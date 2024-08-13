import numpy as np
import ray
import torch

from alg_parameters import *
from mapf_gym import MapfGym
from model import Model
from util import OneEpPerformance, BatchValues, PerfDict


@ray.remote(num_cpus=1, num_gpus=SetupParameters.NUM_GPU / (TrainingParameters.N_ENVS + 1))
class Runner(object):
    """sub-process used to collect experience"""

    def __init__(self, env_id):
        """initialize model0 and environment"""
        self.ID = env_id
        
        self.local_device = torch.device('cuda') if SetupParameters.USE_GPU_LOCAL else torch.device('cpu')
        self.local_model = Model(env_id, self.local_device)

    def run(self, weights):
        return episodeRun(model=self.local_model, weights=weights, eval=False)
    
def episodeRun(model:Model, weights = None, eval = False ):
    assert(eval or weights!=None)
    env = MapfGym()

    """run multiple steps and collect data for reinforcement learning"""
    with torch.no_grad():

        if not eval:
            model.set_weights(weights)
            mb = BatchValues()
        else:
            episodeFrames = []

        performanceDict = PerfDict()
        oneEpisodePerformance = OneEpPerformance()

        svo_rewardList = list()
        action_rewardList = list()
        rewardList = list()
        doneList = list()
        done = False

        for _ in range(TrainingParameters.N_STEPS):

            obs, vector, svo, comms_index = env.getAllObservations()

            if eval:
                episodeFrames.append(env._render())
                actions, pre_block, _, _, _, svo_output = model.evaluate(obs, vector, svo, comms_index)
            else:
                mb.observations.append(obs)
                mb.vector.append(vector)
                mb.svo.append(svo)
                mb.comms_index.append(comms_index)

                actions, ps, values, pre_block, _, svo_output = model.step(obs, vector, svo, comms_index)
                mb.values.append(values)
                mb.ps.append(ps)
                doneList.append(done)
            

            actionStatus, fixedActions = env.getActionStatus(actions, svo_output)

            oneEpisodePerformance.invalid += len(env.getStaticColl(actionStatus))
            svo_post_rewards, action_post_rewards, baseRewards, blockings, leaveGoals, numCollide = env.calculateReward(actions, actionStatus)

            oneEpisodePerformance.block+=np.sum(blockings)
            oneEpisodePerformance.numLeaveGoal += np.sum(leaveGoals)
            oneEpisodePerformance.numCollide+= np.sum(numCollide)
            oneEpisodePerformance.numStep+=1

            for i in range(EnvParameters.N_AGENTS):
                    if (pre_block[i] < 0.5) == blockings[:, i]:
                        oneEpisodePerformance.wrongBlocking += 1
        
            if not eval:
                mb.svo_exe.append(env.getAgentsSVOexe())
                mb.trainValid.append(env.getTrainValid(actions, actionStatus))
                mb.actions.append(actions)
                mb.blocking.append(blockings)
                svo_rewardList.append(svo_post_rewards)
                action_rewardList.append(action_post_rewards)
                rewardList.append(baseRewards)
            
            oneEpisodePerformance.episodeReward += np.sum(baseRewards)

            goalsReached, done = env.jointStep(fixedActions)
            if(done or ((oneEpisodePerformance.numStep+1)%EnvParameters.EPISODE_LEN==0)):
                    done = True

            if oneEpisodePerformance.numStep == EnvParameters.EPISODE_LEN // 2:
                performanceDict.Half_goals.append(np.sum(goalsReached))

            oneEpisodePerformance.maxGoals = max(oneEpisodePerformance.maxGoals, np.sum(goalsReached))


            if done:
                performanceDict.__update__(oneEpisodePerformance, np.sum(goalsReached))
                if eval:
                    episodeFrames.append(env._render()) #append frame to gif
                    break
                else:                        
                    oneEpisodePerformance = OneEpPerformance()

                    env = MapfGym()                                                          
                    done = True

        if not eval:
            mb.observations = np.concatenate(mb.observations, axis=0)
            mb.vector = np.concatenate(mb.vector, axis=0)
            mb.svo = np.concatenate(mb.svo, axis=0)
            mb.svo_exe = np.asarray(mb.svo_exe, dtype=np.int64)
            mb.comms_index = np.concatenate(mb.comms_index, axis=0)
            svo_rewardList = np.concatenate(svo_rewardList, axis=0)
            action_rewardList = np.concatenate(action_rewardList, axis=0)
            rewardList = np.concatenate(rewardList, axis=0)
            mb.values = np.squeeze(np.concatenate(mb.values, axis=0), axis=-1)
            mb.actions = np.asarray(mb.actions, dtype=np.int64)
            mb.ps = np.stack(mb.ps)
            doneList = np.asarray(doneList, dtype=np.bool_)
            mb.trainValid = np.stack(mb.trainValid)
            mb.blocking = np.concatenate(mb.blocking, axis=0)

            last_values  = np.squeeze(model.value(obs, vector, svo, comms_index))

            # calculate advantages
            mb_advs_svo = np.zeros_like(svo_rewardList)
            mb_advs_action = np.zeros_like(action_rewardList)
            mb_advs = np.zeros_like(rewardList)
            last_gaelam_svo = 0
            last_gaelam_action = 0
            last_gaelam = 0
            for t in reversed(range(TrainingParameters.N_STEPS)):
                if t == TrainingParameters.N_STEPS - 1:
                    next_nonterminal = 1.0 - done
                    next_values = last_values
                else:
                    next_nonterminal = 1.0 - doneList[t + 1]
                    next_values= mb.values[t + 1]

                delta_svo = np.subtract(np.add(svo_rewardList[t], TrainingParameters.GAMMA * next_nonterminal *
                                               next_values), mb.values[t])
                delta_action = np.subtract(np.add(action_rewardList[t], TrainingParameters.GAMMA * next_nonterminal *
                                                  next_values), mb.values[t])
                delta = np.subtract(np.add(rewardList[t], TrainingParameters.GAMMA * next_nonterminal *
                                            next_values), mb.values[t])

                mb_advs_svo[t] = last_gaelam_svo = np.add(delta_svo, TrainingParameters.GAMMA * TrainingParameters.LAM
                                                          * next_nonterminal * last_gaelam_svo)
                mb_advs_action[t] = last_gaelam_action = np.add(delta_action,
                                                                TrainingParameters.GAMMA * TrainingParameters.LAM
                                                                * next_nonterminal * last_gaelam_action)
                mb_advs[t] = last_gaelam = np.add(delta, TrainingParameters.GAMMA * TrainingParameters.LAM
                                                        * next_nonterminal * last_gaelam)
            mb.returns_svo = np.add(mb_advs_svo, mb.values)
            mb.returns_action = np.add(mb_advs_action, mb.values)
            mb.returns = np.add(mb_advs, mb.values)
            
    if eval:
        return performanceDict, episodeFrames
    else:
        return mb, performanceDict
