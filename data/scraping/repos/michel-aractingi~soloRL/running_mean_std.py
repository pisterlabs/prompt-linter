import numpy as np
#from PyTorchAgents.envs import VecEnvWrapper
from abc import ABC, abstractmethod


##################
#class and function taken from openAI baselines repo:
#https://github.com/openai/baselines/blob/master/baselines/common/running_mean_std.py
#################

class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, np.float32)
        self.var = np.ones(shape, np.float32)
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)

def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count

class VecEnvWrapper(ABC):
    """
    An environment wrapper that applies to an entire batch
    of environments at once.
    """

    def __init__(self, venv):
        self.venv = venv
        self.nenvs = venv.nenvs

    def step_(self, actions):
        try:
            return self.venv.step(actions)
        except AttributeError:
            self.step_async(actions)
            return self.step_wait()

    @abstractmethod
    def reset(self):
        pass

    def close(self):
        return self.venv.close()

    def render(self, mode='human'):
        return self.venv.render(mode=mode)

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError("attempted to get missing private attribute '{}'".format(name))
        return getattr(self.venv, name)

class VecNormalize(VecEnvWrapper):
    """
    A vectorized wrapper that normalizes the observations
    and returns from an environment.
    """

    def __init__(self, venv, ob=True, ret=True, clipob=10., cliprew=10., gamma=0.99, epsilon=1e-8):

        VecEnvWrapper.__init__(self, venv)
        self.venv = venv
        self.ob_rms = RunningMeanStd(shape=self.observation_space.shape) if ob else None
        self.ret_rms = RunningMeanStd(shape=()) if ret else None
        self.clipob = clipob
        self.cliprew = cliprew
        self._ret_shape = (self.venv.nenvs,)
        if isinstance(self.action_space, list):
            self._ret_shape = (self.venv.nenvs, len(self.action_space))
        self.ret = np.zeros(self._ret_shape)
        self.gamma = gamma
        self.epsilon = epsilon

        self.training = True

    def step(self, actions):
        obs, rews, news, infos = self.step_(actions)
        self.ret = self.ret * self.gamma + rews
        obs = self._obfilt(obs)

        if self.ret_rms:
            if self.training:
                self.ret_rms.update(self.ret)
            rews = np.clip(rews / np.sqrt(self.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew)
        self.ret[news] = 0.

        return obs, rews, news, infos

    def _obfilt(self, obs):
        if self.ob_rms:
            if self.training:
                self.ob_rms.update(obs)
            obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
            return obs
        else:
            return obs

    def reset(self):
        self.ret = np.zeros(self._ret_shape)
        obs = self.venv.reset()
        return self._obfilt(obs)

    def eval(self):
        self.training = False

    def __len__(self):
        return self.nenvs
