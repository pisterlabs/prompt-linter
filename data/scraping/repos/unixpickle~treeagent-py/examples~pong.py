"""
Learn Pong-v0 from OpenAI Gym.
"""

import os

from anyrl.envs import batched_gym_env
from anyrl.envs.wrappers import DownsampleEnv, FrameStackEnv, GrayscaleEnv
from anyrl.rollouts import TruncatedRoller
from anyrl.spaces import gym_space_distribution, gym_space_vectorizer
import gym
import tensorflow as tf

from treeagent import ActorCritic, A2C

POLICY_STEP = 0.1
VALUE_STEP = 0.1
NUM_STEPS = 16
MIN_LEAF = 400
TARGET_KL = 0.01
HORIZON = 512
NUM_WORKERS = 8
FEATURE_FRAC = 0.01

def learn_pong():
    """Train an agent."""
    env = batched_gym_env([make_single_env] * NUM_WORKERS)
    try:
        agent = ActorCritic(gym_space_distribution(env.action_space),
                            gym_space_vectorizer(env.observation_space))
        with tf.Session() as sess:
            a2c = A2C(sess, agent, target_kl=TARGET_KL)
            roller = TruncatedRoller(env, agent, HORIZON)
            total_steps = 0
            rewards = []
            print("Training... Don't expect progress for ~400K steps.")
            while True:
                with agent.frozen():
                    rollouts = roller.rollouts()
                for rollout in rollouts:
                    total_steps += rollout.num_steps
                    if not rollout.trunc_end:
                        rewards.append(rollout.total_reward)
                agent.actor.extend(a2c.policy_update(rollouts, POLICY_STEP, NUM_STEPS,
                                                     min_leaf=MIN_LEAF,
                                                     feature_frac=FEATURE_FRAC))
                agent.critic.extend(a2c.value_update(rollouts, VALUE_STEP, NUM_STEPS,
                                                     min_leaf=MIN_LEAF,
                                                     feature_frac=FEATURE_FRAC))
                if rewards:
                    print('%d steps: mean=%f' %
                          (total_steps, sum(rewards[-10:]) / len(rewards[-10:])))
                else:
                    print('%d steps: no episodes complete yet' % total_steps)
    finally:
        env.close()

def make_single_env():
    """Make a preprocessed gym.Env."""
    env = gym.make('PongNoFrameskip-v4')
    return FrameStackEnv(GrayscaleEnv(DownsampleEnv(env, 2)), num_images=4)

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    learn_pong()
