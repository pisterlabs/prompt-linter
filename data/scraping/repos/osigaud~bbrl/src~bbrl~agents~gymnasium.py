# coding=utf-8
#
# Copyright © Sorbonne University
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import warnings
from abc import ABC
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import logging
import numpy as np
import torch
from torch import nn, Tensor
import gymnasium as gym
from gymnasium import Env, Space, Wrapper, make
from gymnasium.core import ActType, ObsType
from gymnasium.vector import VectorEnv
from gymnasium.wrappers import AutoResetWrapper
from gymnasium.wrappers.monitoring.video_recorder import VideoRecorder

from bbrl import SeedableAgent, SerializableAgent, TimeAgent, Agent
from bbrl.workspace import Workspace


def make_env(env_name, autoreset=False, wrappers: List = [], **kwargs):
    """Utility function to create an environment

    Other parameters are forwarded to the gymnasium `make`

    :param env_name: The environment name
    :param wrappers: Wrappers applied to the base environment **in the order
        they are provided**: that is, if wrappers=[W1, W2], the environment
        (before the optional auto-wrapping) will be W2(W1(base_env))
    :param autoreset: if True, wrap the environment into an AutoResetWrapper,
        defaults to False
    """
    env = make(env_name, **kwargs)
    for wrapper in wrappers:
        env = wrapper(env)

    if autoreset:
        env = AutoResetWrapper(env)
    return env


def record_video(env: Env, agent: Agent, path: str):
    """Record a video for a given gymnasium environment and a BBRL agent

    :param env: The environment (created with `render_mode="rgb_array"`)
    :param agent: The BBRL agent
    :param path: The path of the video
    """

    # Creates the containing folder if needed
    path = Path(path)

    path.parent.mkdir(exist_ok=True, parents=True)

    with torch.no_grad():
        workspace = Workspace()
        obs, _ = env.reset()
        workspace.set("env/env_obs", 0, torch.Tensor(obs).unsqueeze(0))
        t = 0
        done = False
        video_recorder = VideoRecorder(env, str(path.resolve()), enabled=True)
        video_recorder.capture_frame()

        while not done:
            workspace.set("env/env_obs", t, torch.Tensor(obs).unsqueeze(0))
            agent(t=t, workspace=workspace)
            action = workspace.get("action", t).squeeze(0).numpy()
            obs, reward, terminated, truncated, info = env.step(action)
            video_recorder.capture_frame()
            done = terminated or truncated
            t += 1

        video_recorder.close()


def _convert_action(action: Union[Dict,Tensor]) -> Union[int, np.ndarray]:
    if isinstance(action, dict):
        return {key: _convert_action(value) for key, value in action.items()}
    if len(action.size()) == 0:
        action = action.item()
        assert isinstance(action, int)
    else:
        action = np.array(action.tolist())
    return action


def _format_frame(
    frame: Union[Dict[str, Tensor], List[Tensor], np.ndarray, Tensor, bool, int, float]
) -> Union[Tensor, Dict[str, Tensor]]:
    if isinstance(frame, Dict):
        r = {}
        for k in frame:
            r[k] = _format_frame(frame[k])
        return r
    elif isinstance(frame, List):
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
    elif isinstance(frame, Tensor):
        return frame.unsqueeze(0)
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


def _torch_type(d: Dict[str, Tensor]) -> Dict[str, Tensor]:
    return {k: d[k].float() if torch.is_floating_point(d[k]) else d[k] for k in d}


def _torch_cat_dict(d: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
    r = {}
    for k in d[0]:
        r[k] = torch.cat([dd[k] for dd in d], dim=0)
    return r


class GymAgent(TimeAgent, SeedableAgent, SerializableAgent, ABC):
    default_seed = 42

    def __init__(
        self,
        *args,
        input_string: str = "action",
        output_string: str = "env/",
        reward_at_t: bool = False,
        include_last_state: bool = True,
        **kwargs,
    ):
        """Initialize the generic GymAgent

        :param input_string: The name of the variable containing the action,
            defaults to "action"
        :param output_string: The prefix for variables set by the environment,
            defaults to "env/"
        :param reward_at_t: The reward for transitioning from $s_t$ to $s_{t+1}$
            is $r_t$ if reward_at_t is True, and $r_{t+1}$ otherwise (default
            False).
        :param include_last_state: By default (False), the final state is not
            included when using an auto-reset environment. Setting to True allows
            to preserve it.
        """
        super().__init__(*args, **kwargs)

        self.reward_at_t = reward_at_t
        self.include_last_state = include_last_state
        self.ghost_params: nn.Parameter = nn.Parameter(torch.randn(()))

        self.input: str = input_string
        self.output: str = output_string
        self._timestep_from_reset: int = 0
        self._nb_reset: int = 1

        self.observation_space: Optional[Space[ObsType]] = None
        self.action_space: Optional[Space[ActType]] = None

    def forward(self, t: int, *args, **kwargs) -> None:
        if self._seed is None:
            self.seed(self.default_seed)
        if t == 0:
            self._timestep_from_reset = 1
            self._nb_reset += 1
        else:
            self._timestep_from_reset += 1

    def set_obs(self, observations: Dict[str, Tensor], t: int) -> None:
        for k in observations:
            obs = observations[k].to(self.ghost_params.device)
            if self.reward_at_t and k in ["reward", "cumulated_reward"]:
                if t > 0:
                    self.set(
                        (self.output + k, t - 1),
                        obs,
                    )

                # Just use 0 for reward at $t$ for now
                if k == "reward":
                    obs = torch.zeros_like(obs)
            try:
                self.set(
                    (self.output + k, t),
                    obs,
                )
            except Exception:
                logging.error("Error while setting %s", self.output + k)
                raise

    def get_observation_space(self) -> Space[ObsType]:
        """Return the observation space of the environment"""
        if self.observation_space is None:
            raise ValueError("The observation space is not defined")
        return self.observation_space

    def get_action_space(self) -> Space[ActType]:
        """Return the action space of the environment"""
        if self.action_space is None:
            raise ValueError("The action space is not defined")
        return self.action_space

    def get_obs_and_actions_sizes(self) -> Union[int, Tuple[int]]:
        obs_space = self.get_observation_space()
        act_space = self.get_action_space()

        def parse_space(space):
            if len(space.shape) > 0:
                if len(space.shape) > 1:
                    warnings.warn(
                        "Multi dimensional space, be careful, a tuple (shape) "
                        "is returned, maybe youd like to flatten or simplify it first"
                    )
                    return space.shape
                return space.shape[0]
            else:
                return space.n

        return parse_space(obs_space), parse_space(act_space)

    def is_continuous_action(self):
        return isinstance(self.action_space, gym.spaces.Box)

    def is_discrete_action(self):
        return isinstance(self.action_space, gym.spaces.Discrete)

    def is_continuous_state(self):
        return isinstance(self.observation_space, gym.spaces.Box)

    def is_discrete_state(self):
        return isinstance(self.observation_space, gym.spaces.Discrete)


class ParallelGymAgent(GymAgent):
    """Create an Agent from a gymnasium environment

    To create an auto-reset ParallelGymAgent, use the gymnasium
    `AutoResetWrapper` in the make_env_fn
    """

    def __init__(
        self,
        make_env_fn: Callable[[Optional[Dict[str, Any]]], Env],
        num_envs: int,
        make_env_args: Union[Dict[str, Any], None] = None,
        *args,
        **kwargs,
    ):
        """Create an agent from a Gymnasium environment

        Args:
            make_env_fn ([function that returns a gymnasium.Env]): The function
            to create a single gymnasium environments

            num_envs ([int]): The number of environments to create, defaults to
            1

            make_env_args (dict): The arguments of the function that creates a
            gymnasium.Env

            input_string (str, optional): [the name of the action variable in
            the workspace]. Defaults to "action".

            output_string (str, optional): [the output prefix of the
            environment]. Defaults to "env/".
        """
        super().__init__(*args, **kwargs)
        assert num_envs > 0, "n_envs must be > 0"

        self.make_env_fn: Callable[[], Env] = make_env_fn
        self.num_envs: int = num_envs

        self.envs: List[Env] = []
        self.cumulated_reward: Dict[int, float] = {}

        self._timestep: Tensor
        self._is_autoreset: bool = False
        self._last_frame = [None for _ in range(num_envs)]

        args: Dict[str, Any] = make_env_args if make_env_args is not None else {}
        self._initialize_envs(num_envs=num_envs, make_env_args=args)

    def _initialize_envs(self, num_envs: int, make_env_args: Dict[str, Any]):
        self.envs = [self.make_env_fn(**make_env_args) for _ in range(num_envs)]
        self._timestep = torch.zeros(len(self.envs), dtype=torch.long)
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

        # Check if we have an autoreset wrapper somewhere
        _env = self.envs[0]
        while isinstance(_env, Wrapper) and not self._is_autoreset:
            self._is_autoreset = isinstance(_env, AutoResetWrapper)
            _env = _env.env

        if not self._is_autoreset:
            # Do not include last state if not auto-reset
            self.include_last_state = False

    @staticmethod
    def _flatten_value(value: Dict[str, Any]):
        """Flatten nested dict structures with concatenating keys"""
        ret = {}
        for key, value in value.items():
            if isinstance(value, Tensor):
                ret[key] = value
            elif isinstance(value, dict):
                for subkey, subvalue in ParallelGymAgent._flatten_value(value).items():
                    ret[f"{key}/{subkey}"] = subvalue
            else:
                raise ValueError(
                    f"Observation component must be a torch.Tensor or a dict, not {type(observation)}"
                )                

        return ret

    @staticmethod
    def _format_frame(frame):
        observation = _format_frame(frame)

        if isinstance(observation, Tensor):
            return {"env_obs": observation}

        if isinstance(observation, dict):
            return {f"env_obs/{key}": value for key, value in ParallelGymAgent._flatten_value(observation).items()}

        raise ValueError(
            f"Observation must be a torch.Tensor or a dict, not {type(observation)}"
        )

    def _format_obs(
        self, k: int, obs, info, *, terminated=False, truncated=False, reward=0
    ):
        observation: Union[Tensor, Dict[str, Tensor]] = ParallelGymAgent._format_frame(
            obs
        )

        done = terminated or truncated

        if done and self.include_last_state:
            # Create a new frame to be inserted after this step,
            # containing the first observation of the next episode
            self._last_frame[k] = {
                **observation,
                "terminated": torch.tensor([False]),
                "truncated": torch.tensor([False]),
                "done": torch.tensor([False]),
                "reward": torch.tensor([0]).float(),
                "cumulated_reward": torch.tensor([0]).float(),
                "timestep": torch.tensor([0]),
            }
            # Use the final observation instead
            observation = ParallelGymAgent._format_frame(info["final_observation"])

        ret: Dict[str, Tensor] = {
            **observation,
            "terminated": torch.tensor([terminated]),
            "truncated": torch.tensor([truncated]),
            "done": torch.tensor([done]),
            "reward": torch.tensor([reward]).float(),
            "cumulated_reward": torch.tensor([self.cumulated_reward[k]]),
            "timestep": torch.tensor([self._timestep[k]]),
        }

        # Resets the cumulated reward and timestep
        if done and self._is_autoreset:
            self.cumulated_reward[k] = 0.0
            if self._is_autoreset and self.include_last_state:
                self._timestep[k] = 0
            else:
                self._timestep[k] = 1

        return _torch_type(ret)

    def _reset(self, k: int) -> Dict[str, Tensor]:
        """Resets the kth environment

        :param k: The environment index
        :raises ValueError: if the returned observation is not a torch Tensor or
            a dict
        :return: The first observation
        """
        env: Env = self.envs[k]

        self._timestep[k] = 0
        self.cumulated_reward[k] = 0.0

        # Computes a new seed for this environment
        s: int = self._timestep_from_reset * self.num_envs * self._nb_reset * self._seed
        s += (k + 1) * (self._timestep[k].item() + 1 if self._is_autoreset else 1)

        return self._format_obs(k, *env.reset(seed=s))

    def _step(self, k: int, action: Tensor):
        env = self.envs[k]

        action: Union[int, np.ndarray[int]] = _convert_action(action)
        obs, reward, terminated, truncated, info = env.step(action)

        self._timestep[k] += 1
        self.cumulated_reward[k] += reward

        return self._format_obs(
            k, obs, info, terminated=terminated, truncated=truncated, reward=reward
        )

    def forward(self, t: int = 0, **kwargs) -> None:
        """Do one step by reading the `action` at t-1
        If t==0, environments are reset
        If render is True, then the output of env.render() is written as env/rendering
        """
        super().forward(t, **kwargs)

        observations = []
        if t == 0:
            for k, env in enumerate(self.envs):
                observations.append(self._reset(k))
                self._last_frame[k] = None
        else:
            if self.input in self.workspace.variables:
                # Action is a tensor
                action = self.get((self.input, t - 1))
                assert action.size()[0] == self.num_envs, f"Incompatible number of envs ({action.shape[0]} vs {self.num_envs})"
            else:
                # Action is a dictionary
                action = {}
                prefix = f"{self.input}/"
                len_prefix = len(prefix)
                for varname in self.workspace.variables:
                    if not varname.startswith(prefix):
                        continue
                    keys = varname[len_prefix:].split("/")
                    current = action
                    for key in keys[:-1]:
                        current = current.setdefault(key, {})
                    current[keys[-1]] = self.get((varname, t - 1))
                
            def dict_slice(k: int, object):
                if isinstance(object, dict):
                    return {key: dict_slice(k, value) for key, value in object.items()}
                return object[k]
                
            for k, env in enumerate(self.envs):
                if self._last_frame[k] is None:
                    if isinstance(action, dict):
                        frame = self._step(k, dict_slice(k, action))
                    else:
                        frame = self._step(k, action[k])
                else:
                    # Use last frame
                    frame = self._last_frame[k]
                    self._last_frame[k] = None

                observations.append(frame)

                # Reproduce the last frame if over (but with 0 reward)
                if not self._is_autoreset and frame["done"]:
                    self._last_frame[k] = {key: value for key, value in frame.items()}
                    self._last_frame[k]["reward"] = torch.Tensor([0.0])

        self.set_obs(observations=_torch_cat_dict(observations), t=t)


class VecGymAgent(GymAgent):
    """Multi-process

    Use gymnasium VecEnv for multi-process support
    This constrains the environment to be of the auto-reset "type"
    """

    def __init__(
        self,
        make_envs_fn: Callable[[Optional[Dict[str, Any]]], VectorEnv],
        vec_env_args: Optional[Dict[str, Any]] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        args: Dict[str, Any] = vec_env_args or {}
        self.envs: VectorEnv = make_envs_fn(**args)

        self.observation_space = self.envs.single_observation_space
        self.action_space = self.envs.single_action_space

        self.cumulated_reward: Tensor = torch.zeros(self.envs.num_envs)

    def forward(self, t: int, **kwargs) -> None:
        super().forward(t, **kwargs)

        if t == 0:
            s: int = self._seed * self._nb_reset
            obs, infos = self.envs.reset(seed=s)
            terminated = torch.tensor([False] * self.envs.num_envs)
            truncated = torch.tensor([False] * self.envs.num_envs)
            rewards = torch.tensor([0.0] * self.envs.num_envs)
            self.cumulated_reward = torch.zeros(self.envs.num_envs)
        else:
            action = self.get((self.input, t - 1))
            assert (
                action.size()[0] == self.envs.num_envs
            ), "Incompatible number of actions"
            converted_action: Union[int, np.ndarray[int]] = _convert_action(action)
            obs, rewards, terminated, truncated, infos = self.envs.step(
                converted_action
            )
            rewards = torch.tensor(rewards).float()
            terminated = torch.tensor(terminated)
            truncated = torch.tensor(truncated)
            self.cumulated_reward = self.cumulated_reward + rewards

        observation: Union[Tensor, Dict[str, Tensor]] = _format_frame(obs)

        if not isinstance(observation, Tensor):
            raise ValueError("Observation can't be an OrderedDict in a VecEnv")

        ret: Dict[str, Tensor] = {
            "env_obs": observation.squeeze(0),
            "terminated": terminated,
            "truncated": truncated,
            "done": terminated or truncated,
            "reward": rewards,
            "cumulated_reward": self.cumulated_reward,
        }
        self.set_obs(observations=ret, t=t)
