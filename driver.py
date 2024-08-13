import os
import os.path as osp

import numpy as np
import ray
import setproctitle
import torch
import wandb

from alg_parameters import *

if RecordingParameters.WANDB and RecordingParameters.ENTITY=='marmotmapf':
    wandb.login(force = True, key="f03ba093e8210c8999d85ffdea37d56e7292dcbd")

from model import Model
from runner import Runner, episodeRun
from util import set_global_seeds, write_to_wandb, make_gif, BatchValues, PerfDict
# from torch.utils.tensorboard import SummaryWriter

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
ray.init(num_gpus=SetupParameters.NUM_GPU)
print("Welcome to MAPF!\n")


def main():
    """main code"""
    # preparing for training
    if RecordingParameters.RETRAIN:
        restore_path = ''
        net_path_checkpoint = restore_path + "/net_checkpoint.pkl"
        net_dict = torch.load(net_path_checkpoint)

    if RecordingParameters.WANDB:
        if RecordingParameters.RETRAIN:
            wandb_id = None
        else:
            wandb_id = wandb.util.generate_id()
        wandb.init(project=RecordingParameters.EXPERIMENT_PROJECT,
                   name=RecordingParameters.EXPERIMENT_NAME,
                   entity=RecordingParameters.ENTITY,
                   notes=RecordingParameters.EXPERIMENT_NOTE,
                   config=all_args,
                   id=wandb_id,
                   resume='allow')
        print('id is:{}'.format(wandb_id))
        print('Launching wandb...\n')


    setproctitle.setproctitle(
        RecordingParameters.EXPERIMENT_PROJECT + RecordingParameters.EXPERIMENT_NAME + "@" + RecordingParameters.ENTITY)
    set_global_seeds(SetupParameters.SEED)

    # create classes
    global_device = torch.device('cuda') if SetupParameters.USE_GPU_GLOBAL else torch.device('cpu')
    local_device = torch.device('cuda') if SetupParameters.USE_GPU_LOCAL else torch.device('cpu')
    global_model = Model(0, global_device, True)

    if RecordingParameters.RETRAIN:
        global_model.network.load_state_dict(net_dict['model'])
        global_model.net_optimizer.load_state_dict(net_dict['optimizer'])

    envs = [Runner.remote(i + 1) for i in range(TrainingParameters.N_ENVS)]


    if RecordingParameters.RETRAIN:
        curr_steps = net_dict["step"]
        curr_episodes = net_dict["episode"]
        best_perf = net_dict["reward"]
    else:
        curr_steps = curr_episodes = best_perf = 0

    update_done = True
    job_list = []
    last_test_t = -RecordingParameters.EVAL_INTERVAL - 1
    last_model_t = -RecordingParameters.SAVE_INTERVAL - 1
    last_best_t = -RecordingParameters.BEST_INTERVAL - 1
    last_gif_t = -RecordingParameters.GIF_INTERVAL - 1

    # start training
    try:
        while curr_steps < TrainingParameters.N_MAX_STEPS:
            if update_done:
                # start a data collection
                if global_device != local_device:
                    net_weights = global_model.network.to(local_device).state_dict()
                    global_model.network.to(global_device)
                else:
                    net_weights = global_model.network.state_dict()
                net_weights_id = ray.put(net_weights)

                for i, env in enumerate(envs):
                    job_list.append(env.run.remote(net_weights_id))

            # get data from multiple processes
            done_id, job_list = ray.wait(job_list, num_returns=TrainingParameters.N_ENVS)
            update_done = True if job_list == [] else False
            done_len = len(done_id)
            job_results = ray.get(done_id)

            # get reinforcement learning data
            curr_steps += done_len * TrainingParameters.N_STEPS
            mb = BatchValues()
            performance = PerfDict()

            # extract mean batch values
            for results in range(done_len):
                for value in dir(BatchValues()):
                    if not value.startswith('__'):
                        temp = getattr(mb, value)
                        temp.append(getattr(job_results[results][0], value))
                        setattr(mb, value, temp)

                for value in dir(PerfDict()):
                    if not value.startswith('__'):
                        temp = getattr(performance, value)
                        temp.append(getattr(job_results[results][1], value))
                        setattr(performance, value, temp)

                curr_episodes += len(job_results[results][1].Reward) #increment episodes

            for value in dir(BatchValues()):
                if not value.startswith('__'):
                    setattr(mb, value, np.concatenate(getattr(mb, value), axis=0))

            for value in dir(PerfDict()):
                if not value.startswith('__'):
                    # setattr(performance, value, np.concatenate(getattr(performance, value), axis=0))
                    setattr(performance, value, np.nanmean(np.concatenate(getattr(performance, value), axis=0)))


            # training of reinforcement learning
            mb_loss = []
            inds = np.arange(done_len * TrainingParameters.N_STEPS)
            for _ in range(TrainingParameters.N_EPOCHS):
                np.random.shuffle(inds)
                for start in range(0, done_len * TrainingParameters.N_STEPS, TrainingParameters.MINIBATCH_SIZE):
                    end = start + TrainingParameters.MINIBATCH_SIZE
                    mb_inds = inds[start:end]
                    mb_loss.append(global_model.train(mb.observations[mb_inds], mb.vector[mb_inds], mb.svo[mb_inds],
                                                      mb.svo_exe[mb_inds], mb.comms_index[mb_inds],
                                                      mb.returns_svo[mb_inds], mb.returns_action[mb_inds],
                                                      mb.returns[mb_inds],
                                                      mb.values[mb_inds], mb.actions[mb_inds], mb.ps[mb_inds],
                                                      mb.trainValid[mb_inds], mb.blocking[mb_inds]))

            if RecordingParameters.WANDB:
                write_to_wandb(curr_steps, performance, mb_loss, evaluate=False)



            if (curr_steps - last_test_t) / RecordingParameters.EVAL_INTERVAL >= 1.0:
                # if save gif
                if (curr_steps - last_gif_t) / RecordingParameters.GIF_INTERVAL >= 1.0:
                    save_gif = True
                    last_gif_t = curr_steps
                else:
                    save_gif = False

                # evaluate training model
                last_test_t = curr_steps
                with torch.no_grad():
                    evalPerformance, episode_frames = episodeRun(global_model, eval =True)

                    for value in dir(PerfDict()):
                        if not value.startswith('__'):
                            setattr(evalPerformance, value, np.nanmean(getattr(evalPerformance, value), axis=0))

                name = "steps:"+str(curr_steps)+" "
                for i in dir(evalPerformance):
                    if not i.startswith('__'):
                        name+=i
                        name+=":"
                        name+=str(getattr(evalPerformance, i))
                        name+=" "

                if save_gif:
                    if not os.path.exists(RecordingParameters.GIFS_PATH):
                        os.makedirs(RecordingParameters.GIFS_PATH)
                    # print("frames:", len(episode_frames))
                    images = np.array(episode_frames[:-1])
                    make_gif(images,RecordingParameters.GIFS_PATH+"/"+name+'.gif')
                    save_gif = True


                # record evaluation result
                if RecordingParameters.WANDB:
                    # write_to_wandb(curr_steps, greedy_eval_performance_dict, evaluate=True, greedy=True)
                    write_to_wandb(curr_steps, evalPerformance, evaluate=True, greedy=False)
                print(name)
                # save model with the best performance
                if RecordingParameters.RECORD_BEST:
                    if evalPerformance.Reward > best_perf and (
                            curr_steps - last_best_t) / RecordingParameters.BEST_INTERVAL >= 1.0:
                        best_perf = evalPerformance.Reward
                        last_best_t = curr_steps
                        print('Saving best model \n')
                        model_path = osp.join(RecordingParameters.MODEL_PATH, 'best_model')
                        if not os.path.exists(model_path):
                            os.makedirs(model_path)
                        path_checkpoint = model_path + "/net_checkpoint.pkl"
                        net_checkpoint = {"model": global_model.network.state_dict(),
                                          "optimizer": global_model.net_optimizer.state_dict(),
                                          "step": curr_steps,
                                          "episode": curr_episodes,
                                          "reward": best_perf}
                        torch.save(net_checkpoint, path_checkpoint)
            # save model
            if (curr_steps - last_model_t) / RecordingParameters.SAVE_INTERVAL >= 1.0:
                last_model_t = curr_steps
                print('Saving Model !\n')
                model_path = osp.join(RecordingParameters.MODEL_PATH, '%.5i' % curr_steps)
                os.makedirs(model_path)
                path_checkpoint = model_path + "/net_checkpoint.pkl"
                net_checkpoint = {"model": global_model.network.state_dict(),
                                  "optimizer": global_model.net_optimizer.state_dict(),
                                  "step": curr_steps,
                                  "episode": curr_episodes,
                                  "reward": evalPerformance.Reward}
                torch.save(net_checkpoint, path_checkpoint)

    except KeyboardInterrupt:
        print("CTRL-C pressed. killing remote workers")

    # save final model
    print('Saving Final Model !\n')
    model_path = RecordingParameters.MODEL_PATH + '/final'
    os.makedirs(model_path)
    path_checkpoint = model_path + "/net_checkpoint.pkl"
    net_checkpoint = {"model": global_model.network.state_dict(),
                        "optimizer": global_model.net_optimizer.state_dict(),
                        "step": curr_steps,
                        "episode": curr_episodes,
                        "reward": evalPerformance.Reward}
    torch.save(net_checkpoint, path_checkpoint)


    # killing
    for e in envs:
        ray.kill(e)
    if RecordingParameters.WANDB:
        wandb.finish()



if __name__ == "__main__":
    main()

