#!/usr/bin/env python

"""
Train an agent on Sonic using PPO2 from OpenAI Baselines.
"""

import tensorflow as tf

from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
import baselines.ppo2.ppo2 as ppo2
import baselines.ppo2.policies as policies
import baselines.logger as b_logger

from sonic_util import make_env

from polyaxon_helper import get_outputs_path


def main():
    """Run PPO until the environment throws an exception."""
    config = tf.ConfigProto(
        allow_soft_placement=True
    )
    config.gpu_options.allow_growth = True  # pylint: disable=E1101
    tf.Session(config=config).__enter__()

    b_logger.configure(get_outputs_path())

    env = DummyVecEnv([make_env])
    ppo2.learn(policy=policies.CnnPolicy,
               env=env,
               nsteps=4096,
               nminibatches=8,
               lam=0.95,
               gamma=0.99,
               noptepochs=3,
               log_interval=1,
               ent_coef=0.01,
               lr=lambda _: 2e-4,
               cliprange=lambda _: 0.1,
               total_timesteps=int(1e7),
               save_interval=1)


if __name__ == '__main__':
    main()
