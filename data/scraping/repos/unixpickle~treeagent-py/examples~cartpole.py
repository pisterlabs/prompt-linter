"""
Learn CartPole-v0 from OpenAI Gym.
"""

import os

from anyrl.rollouts import BasicRoller, mean_total_reward
from anyrl.spaces import gym_space_distribution, gym_space_vectorizer
import gym
import tensorflow as tf

from treeagent import ActorCritic, A2C

STEP_SIZE = 0.1
VAL_STEP = 0.1
NUM_STEPS = 8

def learn_cartpole():
    """Train an agent."""
    env = gym.make('CartPole-v0')
    try:
        agent = ActorCritic(gym_space_distribution(env.action_space),
                            gym_space_vectorizer(env.observation_space))
        with tf.Session() as sess:
            a2c = A2C(sess, agent, target_kl=0.03)
            roller = BasicRoller(env, agent, min_episodes=8, min_steps=1024)
            while True:
                with agent.frozen():
                    rollouts = roller.rollouts()
                print('mean=%f' % (mean_total_reward(rollouts),))
                agent.actor.extend(a2c.policy_update(rollouts, STEP_SIZE, NUM_STEPS, min_leaf=30))
                agent.critic.extend(a2c.value_update(rollouts, VAL_STEP, NUM_STEPS, min_leaf=30))
    finally:
        env.close()

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    learn_cartpole()
