# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import copy
import torch
from typing import Tuple, Union
import gym
from bbrl.agents.agent import Agent


def _convert_action(action):
    if len(action.size()) == 0:
        action = action.item()
        assert isinstance(action, int)
    else:
        action = np.array(action.tolist())
    return action


def _format_frame(frame):
    if isinstance(frame, dict):
        r = {}
        for k in frame:
            r[k] = _format_frame(frame[k])
        return r
    elif isinstance(frame, list):
        t = torch.tensor(frame).unsqueeze(0)
        if t.dtype == torch.float64 or t.dtype == torch.float32:
            t = t.float()
        else:
            t = t.long()
        return t
    elif isinstance(frame, np.ndarray):
        t = torch.from_numpy(frame).unsqueeze(0)
        if t.dtype == torch.float64 or t.dtype == torch.float32:
            t = t.float()
        else:
            t = t.long()
        return t
    elif isinstance(frame, torch.Tensor):
        return frame.unsqueeze(0)  # .float()
    elif isinstance(frame, bool):
        return torch.tensor([frame]).bool()
    elif isinstance(frame, int):
        return torch.tensor([frame]).long()
    elif isinstance(frame, float):
        return torch.tensor([frame]).float()

    else:
        try:
            # Check if it is a LazyFrame from OpenAI Baselines
            o = torch.from_numpy(frame.__array__()).unsqueeze(0).float()
            return o
        except TypeError:
            assert False


def _torch_type(d):
    nd = {}
    for k in d:
        if d[k].dtype == torch.float64:
            nd[k] = d[k].float()
        else:
            nd[k] = d[k]
    return nd


def _torch_cat_dict(d):
    r = {}
    for k in d[0]:
        r[k] = torch.cat([dd[k] for dd in d], dim=0)
    return r


class GymAgent(Agent):
    """Create an Agent from a gym environment"""

    def __init__(
        self,
        make_env_fn=None,
        make_env_args={},
        n_envs=None,
        seed=None,
        action_string="action",
        output="env/",
    ):
        """Create an agent from a Gym environment

        Args:
            make_env_fn ([function that returns a gym.Env]): The function to create a single gym environments
            make_env_args (dict): The arguments of the function that creates a gym.Env
            n_envs ([int]): The number of environments to create.
            action_string (str, optional): [the name of the action variable in the workspace]. Defaults to "action".
            output (str, optional): [the output prefix of the environment]. Defaults to "env/".
            seed (int): the seed used to initialize the environment
            and each environment will have its own seed]. Defaults to True.
        """
        super().__init__()
        assert n_envs > 0
        self.envs = None
        self.env_args = make_env_args
        self._seed = seed
        assert self._seed is not None, "[GymAgent] seeds must be specified"

        self.n_envs = n_envs
        self.output = output
        self.input = action_string
        self.make_env_fn = make_env_fn
        self.ghost_params = torch.nn.Parameter(torch.randn(()))
        self.timestep = torch.tensor([0 for _ in range(n_envs)])
        self.finished = torch.tensor([True for _ in range(n_envs)])
        self.truncated = torch.tensor([False for _ in range(n_envs)])

        self.envs = [self.make_env_fn(**self.env_args) for _ in range(self.n_envs)]
        for k in range(self.n_envs):
            self.envs[k].seed(self._seed + k)

        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        self.finished = torch.tensor([True for _ in self.envs])
        self.truncated = torch.tensor([True for _ in self.envs])
        self.timestep = torch.tensor([0 for _ in self.envs])
        self.cumulated_reward = {}
        self.last_frame = {}

    def _common_reset(self, k, save_render, render):
        env = self.envs[k]
        self.cumulated_reward[k] = 0.0
        o = env.reset()
        observation = _format_frame(o)

        if isinstance(observation, torch.Tensor):
            observation = {"env_obs": observation}
        else:
            assert isinstance(observation, dict)
        if save_render:
            image = env.render(mode="image").unsqueeze(0)
            observation["rendering"] = image
        elif render:
            env.render(mode="human")

        self.finished[k] = False
        self.truncated[k] = False
        self.timestep[k] = 0

        ret = {
            **observation,
            "done": torch.tensor([False]),
            "truncated": torch.tensor([False]),
            "reward": torch.tensor([0.0]).float(),
            "timestep": torch.tensor([self.timestep[k]]),
            "cumulated_reward": torch.tensor([self.cumulated_reward[k]]).float(),
        }
        return _torch_type(ret), observation

    def _reset(self, k, save_render, render):
        ret, observation = self._common_reset(k, save_render, render)
        self.last_frame[k] = observation
        return ret

    def _make_step(self, env, action, k, save_render, render):
        action = _convert_action(action)

        obs, reward, done, info = env.step(action)
        if "TimeLimit.truncated" in info.keys():
            truncated = info["TimeLimit.truncated"]
        else:
            truncated = False
        self.cumulated_reward[k] += reward
        observation = _format_frame(obs)
        if isinstance(observation, torch.Tensor):
            observation = {"env_obs": observation}
        else:
            assert isinstance(observation, dict)
        if save_render:
            image = env.render(mode="image").unsqueeze(0)
            observation["rendering"] = image
        elif render:
            env.render(mode="human")
        ret = {
            **observation,
            "done": torch.tensor([done]),
            "truncated": torch.tensor([truncated]),
            "reward": torch.tensor([reward]).float(),
            "cumulated_reward": torch.tensor([self.cumulated_reward[k]]),
            "timestep": torch.tensor([self.timestep[k]]),
        }
        return _torch_type(ret), done, truncated, observation

    def _step(self, k, action, save_render, render):
        if self.finished[k]:
            assert k in self.last_frame
            return {
                **self.last_frame[k],
                "done": torch.tensor([True]),
                "truncated": torch.tensor([self.truncated[k]]),
                "reward": torch.tensor([0.0]).float(),
                "cumulated_reward": torch.tensor([self.cumulated_reward[k]]).float(),
                "timestep": torch.tensor([self.timestep[k]]),
            }
        self.timestep[k] += 1
        ret, done, truncated, observation = self._make_step(
            self.envs[k], action, k, save_render, render
        )

        self.last_frame[k] = observation
        if done:
            self.finished[k] = True
            self.truncated[k] = truncated
        return ret

    def set_obs(self, observations, t):
        observations = _torch_cat_dict(observations)
        for k in observations:
            self.set((self.output + k, t), observations[k].to(self.ghost_params.device))

    def forward(self, t=0, save_render=False, render=False, **kwargs):
        """Do one step by reading the `action` at t-1
        If t==0, environments are reset
        If save_render is True, then the output of env.render(mode="image") is written as env/rendering
        """

        if t == 0:
            self.timestep = torch.tensor([0 for _ in self.envs])
            observations = []
            for k, e in enumerate(self.envs):
                obs = self._reset(k, save_render, render)
                observations.append(obs)
            observations = _torch_cat_dict(observations)
            for k in observations:
                self.set(
                    (self.output + k, t), observations[k].to(self.ghost_params.device)
                )
        else:
            assert t > 0
            action = self.get((self.input, t - 1))
            assert action.size()[0] == self.n_envs, "Incompatible number of envs"
            observations = []
            for k, e in enumerate(self.envs):
                obs = self._step(k, action[k], save_render, render)
                observations.append(obs)
            self.set_obs(observations, t)

    def is_continuous_action(self):
        return isinstance(self.action_space, gym.spaces.Box)

    def is_discrete_action(self):
        return isinstance(self.action_space, gym.spaces.Discrete)

    def is_continuous_state(self):
        return isinstance(self.observation_space, gym.spaces.Box)

    def is_discrete_state(self):
        return isinstance(self.observation_space, gym.spaces.Discrete)

    def get_obs_and_actions_sizes(self):
        action_dim = 0
        state_dim = 0
        if self.is_continuous_action():
            action_dim = self.action_space.shape[0]
        elif self.is_discrete_action():
            action_dim = self.action_space.n
        if self.is_continuous_state():
            state_dim = self.observation_space.shape[0]
        elif self.is_discrete_state():
            state_dim = 1  # self.observation_space.n
        return state_dim, action_dim


class AutoResetGymAgent(GymAgent):
    """The same as GymAgent, but with an automatic reset when done is True"""

    def __init__(
        self,
        make_env_fn=None,
        make_env_args={},
        n_envs=None,
        seed=None,
        action_string="action",
        output="env/",
    ):
        """Create an agent from a Gym environment with Autoreset

        Args:
            make_env_fn ([function that returns a gym.Env]): The function to create a single gym environments
            make_env_args (dict): [The arguments of the function that creates a gym.Env]
            n_envs ([int]): [The number of environments to create].
            seed (int, optional): [the seed used to initialize the environment].
            action_string (str, optional): [the name of the action variable in the workspace]. Defaults to "action".
            output (str, optional): [the output prefix of the environment]. Defaults to "env/".
        """
        super().__init__(
            make_env_fn=make_env_fn,
            make_env_args=make_env_args,
            n_envs=n_envs,
            seed=seed,
            action_string=action_string,
            output=output,
        )
        self.is_running = [False for _ in range(self.n_envs)]

    def _reset(self, k, save_render, render):
        self.is_running[k] = True
        ret, _ = self._common_reset(k, save_render, render)
        return ret

    def _step(self, k, action, save_render, render):
        self.timestep[k] += 1
        ret, done, truncated, _ = self._make_step(
            self.envs[k], action, k, save_render, render
        )
        if done:
            self.is_running[k] = False
            self.truncated[k] = truncated
        return ret

    def forward(self, t=0, save_render=False, render=False, **kwargs):
        """
        Perform one step by reading the `action`
        """

        observations = []
        for k, env in enumerate(self.envs):
            if not self.is_running[k] or t == 0:
                observations.append(self._reset(k, save_render, render))
            else:
                assert t > 0
                action = self.get((self.input, t - 1))
                assert action.size()[0] == self.n_envs, "Incompatible number of envs"
                observations.append(self._step(k, action[k], save_render, render))

        self.set_obs(observations, t)


class NoAutoResetGymAgent(GymAgent):
    """The same as GymAgent, named to make sure it is not AutoReset"""

    def __init__(
        self,
        make_env_fn=None,
        make_env_args={},
        n_envs=None,
        seed=None,
        action_string="action",
        output="env/",
    ):
        super().__init__(
            make_env_fn=make_env_fn,
            make_env_args=make_env_args,
            n_envs=n_envs,
            seed=seed,
            action_string=action_string,
            output=output,
        )


# --------------------------- part added from the code of Folco Bertini and Nikola Matevski --------------------


class RunningMeanStd:
    def __init__(self, epsilon: float = 1e-4, shape: Tuple[int, ...] = ()):
        """
        Calulates the running mean and std of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
        :param epsilon: helps with arithmetic issues
        :param shape: the shape of the data stream's output
        """
        self.mean = torch.zeros(shape, dtype=torch.float32)
        self.var = torch.ones(shape, dtype=torch.float32)
        self.count = epsilon
        self.steps = 0
        self.update_threshold = 4
        self.buffer = None

    def copy(self) -> "RunningMeanStd":
        """
        :return: Return a copy of the current object.
        """
        new_object = RunningMeanStd(shape=self.mean.shape)
        new_object.mean = copy.deepcopy(self.mean)
        new_object.var = copy.deepcopy(self.var)
        new_object.count = float(self.count)
        return new_object

    def combine(self, other: "RunningMeanStd") -> None:
        """
        Combine stats from another ``RunningMeanStd`` object.
        :param other: The other object to combine with.
        """
        self.update_from_moments(other.mean, other.var, other.count)

    def update(self, arr: torch.Tensor) -> None:
        self.steps += 1
        if self.buffer is None:
            self.buffer = copy.deepcopy(arr)
        else:
            self.buffer = torch.concat((self.buffer, copy.deepcopy(arr)), dim=0)
        if self.steps > self.update_threshold:
            batch_mean = torch.mean(self.buffer, dim=0)
            batch_var = torch.var(self.buffer, dim=0)
            batch_count = self.buffer.size(0)
            self.steps = 0
            self.buffer = None
            self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(
        self,
        batch_mean: torch.Tensor,
        batch_var: torch.Tensor,
        batch_count: Union[int, float],
    ) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = (
            m_a
            + m_b
            + torch.square(delta)
            * self.count
            * batch_count
            / (self.count + batch_count)
        )
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count


# not checked
class NormalizedNoAutoResetGymAgent(NoAutoResetGymAgent):
    """The same as AutoResetGymAgent, but normalizes observations"""

    def __init__(
        self,
        make_env_fn=None,
        make_env_args={},
        n_envs=None,
        seed=None,
        action_string="action",
        output="env/",
        obs_rms=None,
    ):
        super().__init__(
            make_env_fn=make_env_fn,
            make_env_args=make_env_args,
            n_envs=n_envs,
            seed=seed,
            action_string=action_string,
            output=output,
        )
        self.clip_obs = 10.0
        self.epsilon = 1e-08
        self.obs_rms = obs_rms

    def normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """
        Helper to normalize observation.
        :param obs:
        :return: normalized observation
        """
        obs_ = copy.deepcopy(obs)
        obs_rms = self.obs_rms
        return np.clip(
            (obs_ - obs_rms.mean) / np.sqrt(obs_rms.var + self.epsilon),
            -self.clip_obs,
            self.clip_obs,
        )

    def _reset(self, k, save_render, render):
        full_obs, observation = self._common_reset(k, save_render, render)
        self.last_frame[k] = observation

        full_obs["env_obs"] = self.normalize_obs(full_obs["env_obs"])

        return full_obs

    def _step(self, k, action, save_render, render):
        if self.finished[k]:
            assert k in self.last_frame
            rew = _torch_type({"reward": torch.tensor([0.0]).float()})
            return (
                {
                    **self.last_frame[k],
                    "done": torch.tensor([True]),
                    "truncated": torch.tensor([self.truncated[k]]),
                    "cumulated_reward": torch.tensor(
                        [self.cumulated_reward[k]]
                    ).float(),
                    "timestep": torch.tensor([self.timestep[k]]),
                },
                rew,
            )
        self.timestep[k] += 1
        ret, done, truncated, observation = self._make_step(
            self.envs[k], action, k, save_render, render
        )

        self.last_frame[k] = observation
        if done:
            self.finished[k] = True
            self.truncated[k] = truncated

        ret["env_obs"] = self.normalize_obs(ret["env_obs"])
        return ret


# not checked
class NormalizedAutoResetGymAgent(AutoResetGymAgent):
    """The same as AutoResetGymAgent, but normalizes observations"""

    def __init__(
        self,
        make_env_fn=None,
        make_env_args={},
        n_envs=None,
        seed=None,
        action_string="action",
        output="env/",
    ):
        super().__init__(
            make_env_fn=make_env_fn,
            make_env_args=make_env_args,
            n_envs=n_envs,
            seed=seed,
            action_string=action_string,
            output=output,
        )
        self.clip_obs = 10.0
        self.epsilon = 1e-08
        self.obs_rms = RunningMeanStd(shape=self.observation_space.shape)

    def normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """
        Helper to normalize observation.
        :param obs:
        :return: normalized observation
        """
        obs_ = copy.deepcopy(obs)
        obs_rms = self.obs_rms

        return np.clip(
            (obs_ - obs_rms.mean) / np.sqrt(obs_rms.var + self.epsilon),
            -self.clip_obs,
            self.clip_obs,
        )

    def _reset(self, k, save_render, render):
        self.is_running[k] = True
        full_obs, _ = self._common_reset(k, save_render, render)

        self.obs_rms.update(full_obs["env_obs"])
        full_obs["env_obs"] = self.normalize_obs(full_obs["env_obs"])
        return full_obs

    def _step(self, k, action, save_render, render):
        self.timestep[k] += 1
        ret, done, truncated, _ = self._make_step(
            self.envs[k], action, k, save_render, render
        )
        if done:
            self.is_running[k] = False
            self.truncated[k] = truncated

        self.obs_rms.update(ret["env_obs"])
        ret["env_obs"] = self.normalize_obs(ret["env_obs"])

        return ret
