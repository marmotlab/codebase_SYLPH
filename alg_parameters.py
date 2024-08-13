import datetime

""" Hyperparameters """


class EnvParameters:
    N_AGENTS = 8  # number of agents used in training
    N_ACTIONS = 5
    N_SVOs = 10
    EPISODE_LEN = 256  # maximum episode length in training
    FOV_SIZE = 9
    FOV_Heuristic = 5
    WORLD_SIZE = (10, 40)
    OBSTACLE_PROB = (0.0, 0.3)
    ACTION_COST = -0.3
    IDLE_COST = -0.3
    GOAL_REWARD = 0.0
    COLLISION_COST = -2
    BLOCKING_COST = -1
    OVERLAP_DECAY = 0.95
    IMPORTANCE_SVO = 2
    # 2 means random map, 4 means final version
    GROUP = 2.42


class TrainingParameters:
    lr = 1e-5
    GAMMA = 0.95  # discount factor
    LAM = 0.95  # For GAE
    CLIP_RANGE = 0.2
    MAX_GRAD_NORM = 10
    ENTROPY_COEF = 0.01
    VALUE_COEF = 0.08
    POLICY_COEF = 10
    VALID_COEF = 0.5
    BLOCK_COEF = 0.5
    N_EPOCHS = 10
    N_ENVS = 16  # number of processes
    N_MAX_STEPS = 2e7  # maximum number of time steps used in training
    N_STEPS = 2 ** 8  # number of time steps per process per data collection
    MINIBATCH_SIZE = int(2 ** 8)
    DEMONSTRATION_PROB = 0  # imitation learning rate


class NetParameters:
    NET_SIZE = 512
    SVO_C_SIZE = 512
    NUM_CHANNEL = 9  # number of channels of observations -[FOV_SIZE x FOV_SIZEx NUM_CHANNEL]
    GOAL_REPR_SIZE = 12
    VECTOR_LEN = 4  # [dx, dy, d total, action t-1]


class SetupParameters:
    SEED = 1234
    USE_GPU_LOCAL = False
    USE_GPU_GLOBAL = True
    NUM_GPU = 1


class RecordingParameters:
    RETRAIN = False
    WANDB = False
    TENSORBOARD = False
    TXT_WRITER = True
    ENTITY = 'full_blank_1'
    TIME = datetime.datetime.now().strftime('%d-%m-%y%H%M')
    EXPERIMENT_PROJECT = 'full_blank_2'
    EXPERIMENT_NAME = 'full_blank_3'
    EXPERIMENT_NOTE = 'SYLPH training codebase in random map'
    SAVE_INTERVAL = 5e5  # interval of saving model0
    BEST_INTERVAL = 0  # interval of saving model0 with the best performance
    GIF_INTERVAL = 5e5  # interval of saving gif
    EVAL_INTERVAL = TrainingParameters.N_ENVS * TrainingParameters.N_STEPS  # interval of evaluating training model0
    EVAL_EPISODES = 1  # number of episode used in evaluation
    RECORD_BEST = False
    MODEL_PATH = './models' + '/' + EXPERIMENT_PROJECT + '/' + EXPERIMENT_NAME + TIME
    GIFS_PATH = './gifs' + '/' + EXPERIMENT_PROJECT + '/' + EXPERIMENT_NAME + TIME
    SUMMARY_PATH = './summaries' + '/' + EXPERIMENT_PROJECT + '/' + EXPERIMENT_NAME + TIME



all_args = dict()

for i in dir(EnvParameters()):
    if not i.startswith('__'):
        all_args[i] = getattr(EnvParameters, i)

for i in dir(TrainingParameters()):
    if not i.startswith('__'):
        all_args[i] = getattr(TrainingParameters, i)

for i in dir(NetParameters()):
    if not i.startswith('__'):
        all_args[i] = getattr(NetParameters, i)

for i in dir(SetupParameters()):
    if not i.startswith('__'):
        all_args[i] = getattr(SetupParameters, i)
