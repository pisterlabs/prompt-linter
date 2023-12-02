#!/usr/bin/env python

"""
Train an agent on Sonic using PPO2 from OpenAI Baselines.
"""

import tensorflow as tf
from retro_contest.local import make
import numpy as np

from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.atari_wrappers import WarpFrame, FrameStack
from baselines import logger
import baselines.ppo2.ppo2 as ppo2
import baselines.ppo2.policies as policies
from baselines.deepq import utils
import gym_remote.exceptions as gre
import glob
import os

from sonic_util import make_local_env, SonicDiscretizer, RewardScaler

def main(game, state, timesteps = 5000, save_interval = 1, last_dir=None, params_folder = None):
    """Run PPO until the environment throws an exception."""
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True # pylint: disable=E1101
    env = make(game=game, state=state)
    env = make_local_env(env, stack=True, scale_rew=True)

    logger.configure(params_folder, format_strs=['stdout'])

    def env_fn():
        return env

    load_path = None
    if last_dir:
        list_of_params = glob.glob(last_dir + '/checkpoints/*')
        load_path = max(list_of_params, key=os.path.getctime)
        print('Restoring params from ', load_path)

    with tf.Session(config=config):
        # Take more timesteps than we need to be sure that
        # we stop due to an exception.
        ppo2.learn(policy=policies.CnnPolicy,
                   env=DummyVecEnv([env_fn]),
                   nsteps=4096,
                   nminibatches=8,
                   lam=0.95,
                   gamma=0.99,
                   noptepochs=3,
                   log_interval=1,
                   ent_coef=0.01,
                   lr=lambda _: 2e-4,
                   cliprange=lambda _: 0.1,
                   total_timesteps=timesteps,
                   save_interval=save_interval,
                   load_path=load_path)
        # utils.save_state('/home/noob/retro-noob/ppo/params')

        # model = ppo2.Model(policy=policies.CnnPolicy,
                   # ob_space=tmpEnv.observation_space,
                   # ac_space=tmpEnv.action_space,
                   # nbatch_act=1,
                   # nsteps=4096,
                   # nbatch_train=4096 // 4,
                   # ent_coef=0.01,
                   # vf_coef=0.5,
                   # max_grad_norm=0.5)

        # print(tf.trainable_variables())
        # model2 = utils.load_state('/home/noob/retro-noob/ppo/params')


if __name__ == '__main__':
    main()
