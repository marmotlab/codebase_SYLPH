import ray
import torch
from model import Model
from mapf_gym import MapfGym
from alg_parameters import *
from util import OneEpPerformance, make_gif
import numpy as np
import os
import pickle


env_length = 32
test_num_agents = 50
obs_prob_density = 0.2


@ray.remote(num_cpus=1)
def test_model(num_episode):
    restore_path = 'models/sylph'
    net_path_checkpoint = restore_path + "/net_checkpoint.pkl"
    net_dict = torch.load(net_path_checkpoint, map_location=torch.device('cpu'),weights_only=False)
    test_device = torch.device('cpu')
    test_model = Model(0, test_device)
    test_model.network.load_state_dict(net_dict['model'])
    test_model.network.eval()
    with torch.no_grad():
        success_status, makespan, num_reached = test_one_case(model=test_model, test_episode=num_episode)
    return success_status, makespan, num_reached


def test_one_case(model, test_episode):
    print(f"current_episode is {test_episode}")
    with open('./32_32_0.2/{}length_{}agents_{}density.pth'.format(env_length, test_num_agents, obs_prob_density), 'rb') as f:
        env_info = pickle.load(f)
    fully_arrived = 0
    one_ep_step = 0
    num_reached = 0
    oneEpisodePerformance = OneEpPerformance()
    env = MapfGym()
    env.replicate(-1 * env_info[test_episode][2], env_info[test_episode][3], env_info[test_episode][4])

    done = False
    while not done:
        obs, vector, svo, comms_index = env.getAllObservations()
        actions, pre_block, _, _, _, svo_output = model.evaluate(obs, vector, svo, comms_index)
        actionStatus, fixedActions = env.getActionStatus(actions, svo_output)
        oneEpisodePerformance.invalid += len(env.getStaticColl(actionStatus))
        svo_post_rewards, action_post_rewards, baseRewards, blockings, leaveGoals, numCollide = env.calculateReward(
            actions, actionStatus)
        oneEpisodePerformance.block += np.sum(blockings)
        oneEpisodePerformance.numLeaveGoal += np.sum(leaveGoals)
        oneEpisodePerformance.numCollide += np.sum(numCollide)
        oneEpisodePerformance.numStep += 1
        for i in range(EnvParameters.N_AGENTS):
            if (pre_block[i] < 0.5) == blockings[:, i]:
                oneEpisodePerformance.wrongBlocking += 1
        oneEpisodePerformance.episodeReward += np.sum(baseRewards)
        goalsReached, truelly_done = env.jointStep(fixedActions)
        if truelly_done or ((oneEpisodePerformance.numStep + 1) % EnvParameters.EPISODE_LEN == 0):
            done = True
        oneEpisodePerformance.maxGoals = max(oneEpisodePerformance.maxGoals, np.sum(goalsReached))
        if np.sum(goalsReached) == EnvParameters.N_AGENTS:
            fully_arrived = 1
        one_ep_step = oneEpisodePerformance.numStep
        num_reached = oneEpisodePerformance.maxGoals

    return fully_arrived, one_ep_step, num_reached



if __name__ == "__main__":
    total_success = 0
    total_step = 0
    total_reach = 0
    ray.init(num_cpus=20)
    num_runs = 200
    results = ray.get([test_model.remote(i) for i in range(num_runs)])
    for result in results:
        if result[0] == 1:
            total_success += result[0]
        total_step = total_step + result[1]
        total_reach = total_reach + result[2]
    print(f"Map type is random; env size is {env_length}; num_agent is {test_num_agents}; obs_prob is {obs_prob_density}")
    print("The max steps is", EnvParameters.EPISODE_LEN)
    print(f"success rate is: {total_success / num_runs}")
    print("the average step is: ", total_step / num_runs)
    print("the reach rate is: ", total_reach / (test_num_agents * num_runs))



