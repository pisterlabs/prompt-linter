""" Classes wrapping the OpenAI Gym.
    Some / most of them ar shoplifted from OpenAI/baselines
"""
from collections import deque

import torch
import numpy as np
from termcolor import colored as clr
import gym

from wintermute.envs import ALE
from . import transformations as T


__all__ = [
    "TorchWrapper",
    "SqueezeRewards",
    "FrameStack",
    "DoneAfterLostLife",
    "TransformObservations",
    "MaxAndSkipEnv",
    "FireResetEnv",
    "get_wrapped_atari",
]


class TorchWrapper(gym.ObservationWrapper):
    """ From numpy to torch. """

    def __init__(self, env, verbose=False):
        super().__init__(env)

        if verbose:
            print("[Torch Wrapper] for returning PyTorch Tensors.")

    def observation(self, o):
        """ Convert from numpy to torch.
            Also change from (h, w, c*hist) to (batch, hist*c, h, w)
        """
        return torch.from_numpy(o).permute(2, 0, 1).unsqueeze(0).contiguous()


class NoopResetEnv(gym.Wrapper):
    """Sample initial states by taking random number of no-ops on reset.
    No-op is assumed to be action 0.
    """

    def __init__(self, env, noop_max=30, verbose=False):
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"

        if verbose:
            print(
                f"[NoOp Reset Wrapper] for doing up to {noop_max} no-ops ",
                "at the start of each episode.",
            )

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1)
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


class MaxAndSkipEnv(gym.Wrapper):
    """Return only every `skip`-th frame"""

    def __init__(self, env, skip=4, verbose=False):
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip

        if verbose:
            print(f"[MaxAndSkip Wrapper] for returning only every {skip}th ", "frame.")

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info


class SqueezeRewards(gym.RewardWrapper):
    """ Return only the sign of the reward at each step. """

    def __init__(self, env, verbose=True):
        super().__init__(env)
        if verbose:
            print("[Reward Wrapper] for clamping rewards to -+1")

    def reward(self, reward):
        return float(np.sign(reward))


class TransformObservations(gym.ObservationWrapper):
    """ Applies the given transformations on the observations.
    """

    def __init__(self, env, transformations):
        super().__init__(env)
        self.transformations = transformations

        for t in self.transformations:
            try:
                t.update_env_specs(self)
            except AttributeError:
                pass

    def observation(self, observation):
        for t in self.transformations:
            observation = t.transform(observation)
        return observation

    def _reset(self):
        observation = self.env.reset()
        return self.observation(observation)

    def __str__(self):
        r = ""
        for t in self.transformations:
            r += str(t)
        return "\n<{}({})\n{}>".format(type(self).__name__, r, self.env)


class FrameStack(gym.Wrapper):
    """Stack k last frames. """

    def __init__(self, env, k, verbose=False):
        super().__init__(env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        shape = (shp[0], shp[1], shp[2] * k)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=shape, dtype=np.uint8
        )

        if verbose:
            print(f"[FrameStack Wrapper] for stacking the last {k} frames")

    def reset(self):
        observation = self.env.reset()
        for _ in range(self.k):
            self.frames.append(observation)
        return self._get_ob()

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.frames.append(observation)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return np.concatenate(list(self.frames), axis=2)


class DoneAfterLostLife(gym.Wrapper):
    """ Reset the game if one life is lost in multiple lifes games.
        DeepMind trains with this and evaluates without it.
    """

    def __init__(self, env, verbose=True):
        super(DoneAfterLostLife, self).__init__(env)

        self.no_more_lives = True
        self.crt_live = env.unwrapped.ale.lives()
        self.has_many_lives = self.crt_live != 0

        if self.has_many_lives:
            self.step = self._many_lives_step
        else:
            self.step = self._one_live_step

        not_a = clr("not a", attrs=["bold"])
        if verbose:
            print(
                "[DoneAfterLostLife Wrapper]  %s is %s many lives game."
                % (env.env.spec.id, "a" if self.has_many_lives else not_a)
            )

    def reset(self):
        if self.no_more_lives:
            obs = self.env.reset()
            self.crt_live = self.env.unwrapped.ale.lives()
            return obs
        return self.__obs

    def _many_lives_step(self, action):
        obs, reward, done, info = self.env.step(action)
        crt_live = self.env.unwrapped.ale.lives()
        if crt_live < self.crt_live:
            # just lost a live
            done = True
            self.crt_live = crt_live

        if crt_live == 0:
            self.no_more_lives = True
        else:
            self.no_more_lives = False
            self.__obs = obs
        return obs, reward, done, info

    def _one_live_step(self, action):
        return self.env.step(action)


class FireResetEnv(gym.Wrapper):
    def __init__(self, env, verbose=False):
        """Take action on reset for environments that are fixed until firing."""
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == "FIRE"
        assert len(env.unwrapped.get_action_meanings()) >= 3

        if verbose:
            print(f"[FireReset Wrapper] for automatically resetting envs.")

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


def get_wrapped_atari(env_name, mode="training", no_gym=False, **kwargs):
    """ The preprocessing traditionally used by DeepMind on Atari.
    """
    verbose = kwargs["verbose"] if "verbose" in kwargs else True
    hist_len = kwargs["hist_len"] if "hist_len" in kwargs else 4

    if no_gym:
        return ALE(
            env_name,
            kwargs["seed"],
            kwargs.get("device", torch.device("cpu")),
            history_length=hist_len,
            training=(mode == "training"),
        )

    # spalce_invaders to SpaceInvaders
    env_name = "".join([n.capitalize() for n in env_name.split("_")])
    env_name = f"{env_name}NoFrameskip-v4"

    env = gym.make(env_name)
    assert "NoFrameskip" in env.spec.id

    if mode == "training":
        env = NoopResetEnv(env, noop_max=30, verbose=verbose)

    env = MaxAndSkipEnv(env, skip=4, verbose=verbose)

    if mode == "training":
        env = DoneAfterLostLife(env, verbose=verbose)
        env = SqueezeRewards(env, verbose=verbose)

    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env, verbose=verbose)

    env = TransformObservations(
        env,
        [
            T.Downsample(84, 84),
            T.RGB2Y()
            # T.Normalize()
        ],
    )

    if hist_len != 0:
        env = FrameStack(env, hist_len, verbose=verbose)

    is_torch = kwargs["torch_wrapper"] if "torch_wrapper" in kwargs else True
    if is_torch:
        env = TorchWrapper(env, verbose=verbose)

    return env


if __name__ == "__main__":
    pass
