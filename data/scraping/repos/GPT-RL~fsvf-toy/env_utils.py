# Copyright 2022 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for handling the Atari environment."""


import operator
import re
from collections import deque
from dataclasses import dataclass
from functools import reduce
from typing import Any, ClassVar, Optional

import gym
import numpy as np
from art import text2art
from gym import RewardWrapper, Space  # type: ignore
from gym.core import ObservationWrapper
from gym.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete
from gym.wrappers.time_limit import TimeLimit
from gym_minigrid.minigrid import Goal, Grid, MiniGridEnv, MissionSpace
from gym_minigrid.wrappers import ImgObsWrapper, RGBImgObsWrapper
from ppo import seed_rl_atari_preprocessing
from returns.curry import partial
from returns.pipeline import flow, pipe
from rich.console import Console
from rich.text import Text


class RenderWrapper(gym.Wrapper):
    def reset(self, seed: Optional[int] = None):
        self.__action, self.__reward, self.__done = None, None, None
        self.__state = super().reset(seed=seed)
        return self.__state

    def step(self, action):
        self.__action = action
        self.__state, self.__reward, self.__done, i = super().step(action)
        return self.__state, self.__reward, self.__done, i

    def scale_channel(self, channel):
        if isinstance(self.observation_space, Box):
            high = self.observation_space.high.max()
        elif isinstance(self.observation_space, MultiDiscrete):
            high = self.observation_space.nvec.max()
        elif isinstance(self.observation_space, MultiBinary):
            high = 1
        else:
            raise ValueError(f"Unknown observation space {self.observation_space}")

        return 255 * (channel / high)[:3]

    def ascii_of_image(self, image: np.ndarray) -> Text:
        def rows():
            for row in image:
                yield flow(
                    map(
                        pipe(
                            self.scale_channel,
                            np.cast[int],
                            partial(map, str),
                            ",".join,
                            lambda rgb: f"rgb({rgb})",
                            lambda rgb: Text("██", style=rgb),
                        ),
                        row,
                    ),
                    lambda texts: join_text(*texts, joiner=""),
                )

        return join_text(*rows(), joiner="\n")

    def render(self, mode="human", highlight=True, tile_size=...):
        if mode == "human":
            rgb = self.__state
            console.print(self.ascii_of_image(rgb))
            subtitle = ""
            if self.__action is not None:
                if isinstance(self.__action, int):
                    action = self.__action
                    try:
                        action_str = self.Actions(action).name
                    except AttributeError:
                        action_str = str(action)
                elif isinstance(self.__action, str):
                    action_str = self.__action
                else:
                    raise ValueError(f"Unknown action {self.__action}")
                subtitle += f"action={action_str}, "
            subtitle += f"reward={self.__reward}"
            if self.__done:
                subtitle += ", done"
            print(text2art(subtitle.swapcase(), font="com_sen"))
            input("Press enter to continue.")
        else:
            return super().render(mode=mode, highlight=highlight, tile_size=tile_size)


console = Console()


def join_text(*text: Text, joiner: str) -> Text:
    head, *tail = text
    return reduce(operator.add, [head] + [Text(joiner) + t for t in tail])


class ObsGoalWrapper(ObservationWrapper):
    def __init__(self, env: "EmptyEnv"):
        super().__init__(env)

        coord_space = MultiDiscrete(np.array([env.width, env.height]))
        assert isinstance(self.observation_space, Dict)
        self.observation_space = Dict(
            dict(**self.observation_space.spaces, agent=coord_space, goal=coord_space)
        )

    def observation(self, obs):
        assert isinstance(self.env, EmptyEnv)
        return dict(**obs, agent=self.env.agent_pos, goal=self.env.goal_pos)


class FlatObsWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(self.observation_space, Dict)
        self.observation_space = flow(
            self.observation_space.spaces,
            self.get_nvecs,
            np.array,
            MultiDiscrete,
        )

    def get_nvecs(self, spaces: dict[str, Space]):
        agent_space = spaces["agent"]
        goal_space = spaces["goal"]
        assert isinstance(agent_space, MultiDiscrete)
        assert isinstance(goal_space, MultiDiscrete)
        return [
            *agent_space.nvec,
            *goal_space.nvec,
        ]

    def get_observations(self, obs: dict[str, Any]) -> list[np.ndarray]:
        return [obs["agent"], obs["goal"]]

    def observation(self, obs):
        return np.concatenate(self.get_observations(obs))


class FlatObsWithDirectionWrapper(FlatObsWrapper):
    def get_nvecs(self, spaces: dict[str, Space]):
        dir_space = spaces["direction"]
        assert isinstance(dir_space, Discrete)
        return super().get_nvecs(spaces) + [np.array([dir_space.n])]

    def get_observations(self, obs: dict[str, Any]) -> list[np.ndarray]:
        return super().get_observations(obs) + [np.array(obs["direction"])]


class OneHotWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        space = self.observation_space
        self.observation_space = gym.spaces.MultiBinary(
            np.array([*space.nvec.shape, space.nvec.max()])
        )
        self.one_hot = np.eye(space.nvec.max(), dtype=np.int)

    def observation(self, obs):
        return self.one_hot[obs]


class FlattenWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        assert isinstance(self.observation_space, MultiBinary)
        self.observation_space = MultiBinary(int(np.prod(self.observation_space.n)))

    def observation(self, obs):
        return obs.flatten()


class TwoDGridWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        assert hasattr(env, "height")
        assert hasattr(env, "width")
        self.empty = np.zeros((env.height, env.width), dtype=np.int)
        self.observation_space = MultiDiscrete(3 * np.ones((env.height, env.width)))

    def observation(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        grid = np.copy(self.empty)
        grid[tuple(obs["agent"])] = 1
        grid[tuple(obs["goal"])] = 2
        return grid


class EmptyEnv(MiniGridEnv):
    """
    Empty grid environment, no obstacles, sparse reward
    """

    def __init__(self, size=8, agent_start_pos=(1, 1), agent_start_dir=0, **kwargs):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        mission_space = MissionSpace(
            mission_func=lambda: "get to the green goal square"
        )

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            max_steps=4 * size * size,
            # Set this to True for maximum speed
            see_through_walls=True,
            **kwargs,
        )

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal square in the bottom-right corner
        self.goal_pos = self.place_obj(Goal())

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "get to the green goal square"

    def reset(self, seed: Optional[int] = None):
        seed = seed or 0
        return super().reset(seed=seed)


@dataclass
class MyEnv(gym.Env):
    height: int
    width: int
    deltas: ClassVar[np.ndarray] = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
    pattern: ClassVar[str] = r"my-env-(\d+)x(\d+)"

    def __post_init__(self):
        self.observation_space = Dict(
            dict(
                agent=MultiDiscrete(np.array([self.height, self.width])),
                goal=MultiDiscrete(np.array([self.height, self.width])),
            )
        )

    @classmethod
    @property
    def action_space(cls):
        return Discrete(1 + len(cls.deltas))

    def random_pos(self) -> np.ndarray:
        pos = self.np_random.randint(low=0, high=(self.height, self.width))
        assert isinstance(pos, np.ndarray)
        return pos

    def reset(self, **kwargs) -> dict[str, np.ndarray]:
        super().reset(**kwargs)
        self.agent = self.random_pos()
        self.goal = self.random_pos()
        return self.state()

    def state(self) -> dict[str, np.ndarray]:
        return dict(agent=self.agent, goal=self.goal)

    def step(self, action: int) -> tuple[dict[str, np.ndarray], float, bool, dict]:
        r = 0.0
        t = False
        try:
            delta = self.deltas[action]
        except IndexError:
            r = float(all(self.agent == self.goal))
            t = True
            return self.state(), r, t, {}

        agent = self.agent + delta
        self.agent = np.clip(agent, 0, (self.height - 1, self.width - 1))
        return self.state(), r, t, {}

    def render(self, mode: Any = ...) -> None:
        for i in range(self.height):
            for j in range(self.width):
                if all(self.agent == np.array([j, i])):
                    print("A", end="")
                elif all(self.goal == np.array([j, i])):
                    print("G", end="")
                else:
                    print("-", end="")
            print()

        input("Press Enter to continue...")
        return None


class ClipRewardEnv(RewardWrapper):
    """Adapted from OpenAI baselines.

    github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
    """

    def __init__(self, env):
        RewardWrapper.__init__(self, env)

    def reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return np.sign(reward)


class FrameStack:
    """Implements stacking of `num_frames` last frames of the game.
    Wraps an AtariPreprocessing object.
    """

    def __init__(
        self, preproc: seed_rl_atari_preprocessing.AtariPreprocessing, num_frames: int
    ):
        self.preproc = preproc
        self.num_frames = num_frames
        self.frames: deque = deque(maxlen=num_frames)
        assert isinstance(preproc.observation_space, Box)

        def repeat(x):
            return np.repeat(x, num_frames, axis=-1)

        obs_space = preproc.observation_space
        self.observation_space = Box(
            low=repeat(obs_space.low), high=repeat(obs_space.high)
        )
        self.np_random = preproc.environment.np_random

    def reset(self, seed: Optional[int] = None):
        ob = self.preproc.reset(seed=seed)
        for _ in range(self.num_frames):
            self.frames.append(ob)
        return self._get_array()

    def step(self, action: int):
        ob, reward, done, info = self.preproc.step(action)
        self.frames.append(ob)
        return self._get_array(), reward, done, info

    def _get_array(self):
        assert len(self.frames) == self.num_frames
        return np.concatenate(self.frames, axis=-1)


def create_env(env_id: str, test: bool):
    """Create a FrameStack object that serves as environment for the `game`."""
    if env_id == "empty":
        return flow(
            EmptyEnv(size=4, agent_start_pos=None),
            RGBImgObsWrapper,
            ImgObsWrapper,
            RenderWrapper,
        )
    elif re.match(MyEnv.pattern, env_id):
        [(height, width)] = re.findall(MyEnv.pattern, env_id)
        height, width = map(int, (height, width))
        return flow(
            MyEnv(height=height, width=width),
            TwoDGridWrapper,
            OneHotWrapper,
            RenderWrapper,
            partial(TimeLimit, max_episode_steps=10 + height + width),
        )
    elif "NoFrameskip" in env_id:
        return flow(
            gym.make(env_id),
            *([] if test else [ClipRewardEnv]),
            seed_rl_atari_preprocessing.AtariPreprocessing,
            partial(FrameStack, num_frames=4),
            RenderWrapper,
        )
    elif "MiniGrid" in env_id:
        return flow(gym.make(env_id), RGBImgObsWrapper, ImgObsWrapper, RenderWrapper)
    else:
        return gym.make(env_id)


def get_num_actions(game: str):
    """Get the number of possible actions of a given Atari game.

    This determines the number of outputs in the actor part of the
    actor-critic model.
    """
    env = gym.make(game)
    assert isinstance(env.action_space, Discrete)
    return env.action_space.n
