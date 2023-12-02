# Adapted from https://tensorforce.readthedocs.io/en/latest/

import numpy as np

from tensorforce.agents import PPOAgent
from tensorforce.execution import Runner
from tensorforce.contrib.openai_gym import OpenAIGym
import matplotlib.pyplot as plt
import sys

viz = False
if "--show" in sys.argv:
    viz = True

env = OpenAIGym('LunarLander-v2', visualize=viz, monitor="monitor/", monitor_safe=False, monitor_video=True)

training_progress = []

agent = PPOAgent(
    states=env.states,
    actions=env.actions,
    network=[
        dict(type='dense', size=32, activation='relu'),
        dict(type='dense', size=32, activation='relu'),
        dict(type='dense', size=32, activation='relu'),
        dict(type='dense', size=32, activation='relu'),
    ],
    batching_capacity=32,
    step_optimizer=dict(
        type='adam',
        learning_rate=1e-3
    ),
    scope='ppo',
    discount=0.99,
    entropy_regularization=0.01,
    baseline_mode=None,
    baseline=None,
    baseline_optimizer=None,
    gae_lambda=None,
    likelihood_ratio_clipping=0.2,
)

if "--resume" in sys.argv:
    agent.restore_model(directory="models/")

# Create the runner
runner = Runner(agent=agent, environment=env)


# Callback function printing episode statistics
def episode_finished(r):
    print("[{ep}] @ {ts}ts -> \t{reward}".format(ep=r.episode, ts=r.episode_timestep,
                                                 reward=r.episode_rewards[-1]))
    training_progress.append(r.episode_rewards[-1])
    if r.episode % 100 == 0:
        if not viz:
            env.visualize = True
        agent.save_model(directory="models/")
        plt.scatter(range(len(training_progress)), training_progress, s=1)
        plt.title("Lunar Lander Training Progress\n5-layer 32-neurons/layer ReLU")
        plt.xlabel("Episodes")
        plt.ylabel("Reward")
        plt.savefig(fname="training_progress.png")
    else:
        if not viz:
            env.visualize = False
    return True


# Start learning
runner.run(max_episode_timesteps=10000, episode_finished=episode_finished)
runner.close()

# Print statistics
print("Learning finished. Total episodes: {ep}. Average reward of last 100 episodes: {ar}.".format(
    ep=runner.episode,
    ar=np.mean(runner.episode_rewards[-100:]))
)
