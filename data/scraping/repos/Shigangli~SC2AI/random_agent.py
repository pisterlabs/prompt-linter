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

from tensorforce.agents import RandomAgent
from tensorforce.execution import Runner
from tensorforce.contrib.openai_gym import OpenAIGym

import sc2gym
from absl import flags
FLAGS = flags.FLAGS
FLAGS([__file__])


# Create an OpenAIgym environment
env = OpenAIGym('SC2CollectMineralShards-v2', visualize=False)


agent = RandomAgent(
    states_spec=env.states,
    actions_spec=env.actions,
)
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
runner.run(episodes=200, episode_finished=episode_finished)

# Print statistics
print("Learning finished. Total episodes: {ep}. Average reward of last 100 episodes: {ar}.".format(
    ep=runner.episode,
    ar=np.mean(runner.episode_rewards[-100:]))
)
