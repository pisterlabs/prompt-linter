# compile cython modules
import os
os.system('python experience_replay_setup.py build_ext --inplace')

# load dependencies
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras import initializers

import gym

import numpy as np

from deep_q_agents import EpsAnnDQNAgent
from deep_q_networks import DeepQNetwork
from experience_replay import PrioritizedExperienceReplay
from atari_preprocessing import atari_pong_processor, ProcessedAtariEnv
from openai_baseline_wrappers import make_atari, wrap_deepmind
from load_data import LoadAtariHeadData


#create environment
frame_processor = atari_pong_processor
game_id = 'PongNoFrameskip-v4'
env = make_atari(game_id)
env = wrap_deepmind(env)
env = ProcessedAtariEnv(env, frame_processor, reward_processor = lambda x: np.sign(x), neg_reward_terminal = True)

# additional env specific parameters
frame_shape = env.reset().shape
frame_skip = 4
num_stacked_frames = 4
num_actions = env.action_space.n

# replay parameters
batch_size = 32
max_frame_num = 2**20
prioritized_replay = True
prio_coeff = 1.0
is_schedule = [0.0, 0.0, 2500000]
replay_epsilon = 0.001
memory_restore_path = None

# network training parameters
dueling = True
double_q = True
lr_schedule = [[0.00025, 0.00025, 10000000]]
optimizer = Adam
discount_factor = 0.99
n_step = 10
one_step_weight = 1.0/2.0
n_step_weight = 1.0/2.0
expert_weight = 0.0
model_restore_path = None

# network architecture
conv_layers = {'filters': [32, 64, 64, 1024],
               'kernel_sizes': [8, 4, 3, 7],
               'strides': [4, 2, 1, 1],
               'paddings': ['valid' for _ in range(4)],
               'activations': ['relu' for _ in range(4)],
               'initializers': [initializers.VarianceScaling(scale = 2.0) for _ in range(4)],
               'names': ['conv_%i'%(i) for i in range(1,5)]}
dense_layers = None

# exploration parameters
eps_schedule = [[1, 0.1, 50000],
                [0.1, 0.01, 100000],
                [0.01, 0.001, 100000]]

# training session parameters
target_interval = 10000
warmup_steps = 50000
pretrain_steps = None
learning_interval = 4
num_steps = 500000
num_episodes = 300
max_steps_per_episode = 100000
output_freq = 30
save_freq = 30
store_memory = False
save_path = "experiments/pong_standard_experiment/"



# create replay memory
memory = PrioritizedExperienceReplay(frame_shape = frame_shape,
                                     max_frame_num = max_frame_num,
                                     num_stacked_frames = num_stacked_frames,
                                     batch_size = batch_size,
                                     prio_coeff = prio_coeff,
                                     is_schedule = is_schedule,
                                     epsilon = replay_epsilon,
                                     restore_path = memory_restore_path)

# expert memory
expert_memory = None

# create policy network
policy_network = DeepQNetwork(in_shape = (num_stacked_frames, *frame_shape),
                              conv_layers = conv_layers,
                              dense_layers = dense_layers,
                              num_actions = num_actions,
                              optimizer = optimizer,
                              lr_schedule = lr_schedule,
                              dueling = dueling,
                              one_step_weight = one_step_weight,
                              n_step_weight = n_step_weight,
                              expert_weight = expert_weight)


if model_restore_path is not None:
    policy_network.model.load_weights(model_restore_path, by_name = True)

# create target network
target_network = DeepQNetwork(in_shape = (num_stacked_frames, *frame_shape),
                              conv_layers = conv_layers,
                              dense_layers = dense_layers,
                              num_actions = num_actions,
                              optimizer = optimizer,
                              lr_schedule = lr_schedule,
                              dueling = dueling,
                              one_step_weight = one_step_weight,
                              n_step_weight = n_step_weight,
                              expert_weight = expert_weight)

if model_restore_path is not None:
    target_network.model.load_weights(model_restore_path, by_name = True)

# create agent
agent = EpsAnnDQNAgent(env = env,
                       memory = memory,
                       policy_network = policy_network, 
                       target_network = target_network,
                       num_actions = num_actions,
                       frame_shape = frame_shape,
                       discount_factor = discount_factor,
                       save_path = save_path,
                       eps_schedule = eps_schedule,
                       double_q = double_q,
                       n_step = n_step,
                       expert_memory = expert_memory,
                       prioritized_replay = prioritized_replay)


agent.policy_network.model.save(save_path + "/trained_models/initial_model.h5")

# train the agent
agent.train(num_episodes = num_episodes,
            num_steps = num_steps,
            max_steps_per_episode = max_steps_per_episode,
            warmup_steps = warmup_steps,
            pretrain_steps = pretrain_steps,
            target_interval = target_interval,  
            learning_interval = learning_interval,
            output_freq = output_freq,
            save_freq = save_freq,
            store_memory = store_memory)
