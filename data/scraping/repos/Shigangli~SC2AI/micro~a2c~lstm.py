# Copyright 2017 reinforce.io. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np

from tensorforce.agents import PPOAgent
from tensorforce.execution import Runner
from tensorforce.contrib.openai_gym import OpenAIGym

import sc2gym
from absl import flags
FLAGS = flags.FLAGS
FLAGS([__file__])


# Create an OpenAIgym environment
# ReversedAddition-v0
# CartPole-v0
env = OpenAIGym('SC2CollectMineralShards-v2', visualize=False)

# Network as list of layers
network_spec = [
    dict(type='conv2d', size=64),
    dict(type='flatten'),
    dict(type='dense', size=32, activation='relu'),
    dict(type='lstm', size=64)
]

saver_spec = {
    'load': True,
    'file': 'model.ckpt-5410984',
    'directory': './model1',
    'seconds': 18000
}

agent = PPOAgent(
    states_spec=env.states,
    actions_spec=env.actions,
    network_spec=network_spec,
    batch_size=10,
    # Agent
    preprocessing=None,
    exploration=None,
    reward_preprocessing=None,
    saver_spec=saver_spec,
    # BatchAgent
    keep_last_timestep=True,
    # PPOAgent
    step_optimizer=dict(
        type='adam',
        learning_rate=1e-5
    ),
    optimization_steps=10,
    # Model
    scope='ppo',
    discount=0.99,
    # DistributionModel
    distributions_spec=None,
    entropy_regularization=0.01,
    # PGModel
    baseline_mode="states",
    baseline={
        "type": "cnn",
        "conv_sizes": [32],
        "dense_sizes": [32]
    },
    baseline_optimizer={
        "type": "multi_step",
        "optimizer": {
            "type": "adam",
            "learning_rate": 1e-5
        },
        "num_steps": 10
    },
    gae_lambda=0.99,
    normalize_rewards=False,
    # PGLRModel
    likelihood_ratio_clipping=0.2,
    summary_spec=None,
    distributed_spec=None
)
    
print('partially success')

# Create the runner
runner = Runner(agent=agent, environment=env)


# Callback function printing episode statistics
rewards = []
def episode_finished(r):
    print("Finished episode {ep} after {ts} timesteps (reward: {reward})".format(ep=r.episode, ts=r.episode_timestep,
                                                                                 reward=r.episode_rewards[-1]))
    global rewards
    rewards += [r.episode_rewards[-1]]
    return True


# Start learning
runner.run(episodes=60000, episode_finished=episode_finished)

# Print statistics
print("Learning finished. Total episodes: {ep}. Average reward of last 100 episodes: {ar}.".format(
    ep=runner.episode,
    ar=np.mean(runner.episode_rewards[-100:]))
)
