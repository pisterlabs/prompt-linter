#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import json
import numpy as np
from utils.base import Base
from gym_training.envs.training_env import TrainingEnv
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.sac.policies import SACPolicy
from monitor.srv import GuidanceInfoRequest
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise


class Learner(Base):
    def __init__(self):
        super(Learner, self).__init__(__file__)
        kwargs = {
            'experiment_series': self.prms['experiment_series'],
            'experiment': self.prms['experiment'],
            'arm': self.prms['arm'],
            'angular': self.prms['angular'],
            'penalty_deviation': self.prms['penalty_deviation'],
            'penalty_angular': self.prms['penalty_angular'],
            'time_step_limit': self.prms['time_step_limit'],
            'sigma': self.prms['sigma'],
            'task': self.prms['task'],
            'rand': self.prms['rand'],
            'env_type': self.prms['env_type']
        }
        env = DummyVecEnv([lambda: TrainingEnv(**kwargs)])
        policies = {
            "default": "MlpPolicy",
            "sac_default": "MlpPolicy",
            "ppo_default": "MlpPolicy",
            "td3_default": "MlpPolicy",
        }
        self.time_step_cnt = 0
        model_parameters = self.prms['model_parameters']
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
        if self.loading:
            models = {
                "SAC": SAC.load(self.load_path, env=env),
                "PPO": PPO.load(self.load_path, env=env),
                "TD3": TD3.load(self.load_path, env=env)
            }
        else:
            models = {
                "SAC": SAC(
                    policies[self.prms["policy"]],
                    env,
                    verbose=1,
                    train_freq=int(model_parameters["sac_train_freq"]),
                    tau=float(model_parameters["sac_tau"]),
                    ent_coef=model_parameters["sac_ent_coef"],
                    tensorboard_log=self.log_path
                ),
                "PPO": PPO(
                    policies[self.prms["policy"]],
                    env,
                    verbose=1,
                    tensorboard_log=self.log_path
                ),
                "TD3": TD3(
                    policies[self.prms["policy"]],
                    env,
                    action_noise=action_noise,
                    verbose=1,
                    tensorboard_log=self.log_path
                ),
            }
        self.model = models[self.prms["alg"]]


    def train_model(self):
        self.log(f"Learning started! Model is located at {self.save_path}")
        self.model.learn(total_timesteps=int(self.prms['total_timesteps']), log_interval=4,
                         callback=self.callback)
        try:
            self.model.save(self.save_path)
        except Exception as e:
            print("Model was not saved")
            print(e)

    def callback(self, _locals, _globals):
        """
        :param _locals: local variables from the model
        :param _globals: global variables from the model
        :return: boolean value for whether or not the training should continue
        """
        resp = self.guidance_info.call(GuidanceInfoRequest())

        self.time_step_cnt += 1
        # model is saved every 1000 time steps
        if self.time_step_cnt % 1000 == 0:
            _locals['self'].save(self.save_path)

        if resp.done:
            _locals['self'].save(self.save_path)
            # Stop training.
            return False
        else:
            # Continue training.
            return True


if __name__ == '__main__':
    Learner().train_model()
