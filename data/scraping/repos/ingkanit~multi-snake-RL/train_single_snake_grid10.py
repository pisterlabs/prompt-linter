import numpy as np

from tensorforce.agents import PPOAgent
from tensorforce.execution import Runner
from tensorforce.contrib.openai_gym import OpenAIGym
from OpenAIGym_custom import OpenAIGym_custom
from multi_snake import MultiSnake
from analyze_training_curve import findlastepisode
from handle_args import handle_args

import gym
from gym import spaces

import os
import subprocess as sp

# set environment variable to free gpu device
out = sp.check_output('./free_gpus.sh').split()[0]
if isinstance(out, bytes):
    out = out.decode()
os.environ["CUDA_VISIBLE_DEVICES"] = out

SAVE_INT = 1000
DISPLAY_INT = 100
trainhistdir = 'train_single_snake_grid10/'
save_gif = False
test_model = True
continue_training = False

num_episodes = int(6e5)
episode_offset = 0
max_episode_timesteps = 1000

giffn = trainhistdir + 'video.gif'

# handle command line arguments
test_model, save_gif = handle_args(test_model, save_gif)

if save_gif:
    num_episodes = 3

# Create custom OpenAIgym environment
num_agents = 1
spacing = 22
grid_dim = 10
e = MultiSnake(num_agents=num_agents, num_fruits=3, spacing=spacing, grid_dim=grid_dim, save_gif=save_gif)

env = OpenAIGym_custom(e, "MultiSnake", visualize=test_model)

# Network as list of layers
network_spec = [
    dict(type='dense', size=128, activation='tanh'),
    dict(type='dense', size=128, activation='tanh')
]

agent = PPOAgent(
    states=env.states,
    actions=env.actions,
    network=network_spec,
    update_mode = dict(
        unit='episodes',
        batch_size=10,
        frequency=10
    ),
    memory = dict(
        type='latest',
        include_next_states=False,
        capacity=5000
    ),
    subsampling_fraction=0.1,
    # PPOAgent
    step_optimizer=dict(
        type='adam',
        learning_rate=1e-3
    ),
    optimization_steps=50,
    # Model
    scope='ppo',
    discount=0.99,
    entropy_regularization=0.01,
    # PGModel
    baseline_mode='states',
    baseline=dict(
        type='mlp',
        sizes=[64,64]
        ),
    baseline_optimizer=dict(
        type='multi_step',
        optimizer=dict(
            type='adam',
            learning_rate=1e-3
            ),
        num_steps=5),
    gae_lambda=None,
    # PGLRModel
    likelihood_ratio_clipping=0.2
)

# Create the runner
runner = Runner(agent=agent, environment=env)

# If testing or continue_training, load model parameters
if test_model or continue_training:
    runner.agent.restore_model(trainhistdir)

stat = np.zeros((num_episodes,2))
global_episode_counter = 1

if continue_training:
    episode_offset = findlastepisode(trainhistdir)
    stat[:episode_offset,:] = np.load('{}Reward_stat_{}.npz'.format(trainhistdir,episode_offset))['stat'][:episode_offset,:]

# Callback function printing episode statistics
def episode_finished(r):
    global global_episode_counter
    
    print("Finished episode {ep} after {ts} timesteps (reward: {reward})".format(ep=r.episode, ts=r.episode_timestep,
                                                                                 reward=r.episode_rewards[-1]))
    
    # store current statistics
    if not test_model:
        stat[r.episode-1,:] = [r.episode_timestep,r.episode_rewards[-1]]
                                                                                 
    if not test_model and (r.episode-1) % SAVE_INT == 0:
        r.agent.save_model(trainhistdir)
        np.savez("{}Reward_stat_{}.npz".format(trainhistdir,r.episode),stat=stat)
        print("Model saved.")
                                                                                 
    if global_episode_counter > num_episodes:
        return False
    else:
        global_episode_counter += 1
        return True
   
# Start learning
runner.run(episodes=num_episodes, max_episode_timesteps=max_episode_timesteps, episode_finished=episode_finished)
# runner.close()

# Print statistics
print("Learning finished. Total episodes: {ep}. Average reward of last 100 episodes: {ar}.".format(
    ep=runner.episode,
    ar=np.mean(runner.episode_rewards[-100:]))
)

if not test_model:
    runner.agent.save_model(trainhistdir)
    np.savez("{}Reward_stat_{}.npz".format(trainhistdir,runner.episode),stat=stat)
    print("Model saved.")

if save_gif:
    e.write_gif(giffn)