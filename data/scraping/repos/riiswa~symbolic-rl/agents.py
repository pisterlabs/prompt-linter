import random
from functools import lru_cache

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import gym
import torch
import numpy as np
from gym import Space
from gym.spaces import MultiDiscrete, Discrete
from stable_baselines3.common.base_class import BaseAlgorithm, SelfBaseAlgorithm
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import MaybeCallback

from SymbolicEnv import SymbolicEnv


def get_dims(space: Space):
    """
    Gets the dimensions of a space.

    Args:
        space (Space): A space object.

    Returns:
        list: A list of integers representing the dimensions of the space.

    Raises:
        RuntimeError: If the input `space` is not a `MultiDiscrete` or a `Discrete` type.
    """
    if isinstance(space, MultiDiscrete):
        return [s.n for s in space]
    elif isinstance(space, Discrete):
        return [space.n]
    else:
        raise RuntimeError("Not supported space.")


class QLearningPolicy(BasePolicy):
    """
    Implements a Q-Learning policy for reinforcement learning.

    Attributes:
        observation_space (Space): The observation space of the environment.
        action_space (Space): The action space of the environment.
        qtable (torch.Tensor): The Q-Table storing the Q-values for each state-action pair.

    """
    def __init__(self, observation_space: Space, action_space: Space):
        BasePolicy.__init__(self, observation_space, action_space)
        self.observation_space = observation_space
        self.action_space = action_space

        dims = []
        dims.extend(get_dims(observation_space))
        dims.extend(get_dims(action_space))

        self.qtable = torch.zeros(tuple(dims), requires_grad=False)

    def _predict(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        Predicts the best action to take given the current observation.

        Args:
            observation (torch.Tensor): The current observation of the environment.
            deterministic (bool, optional): Whether to return a deterministic action
                                           (the action with the highest Q-value), or a random action among the actions
                                           with the highest Q-value.
                                           Default is False.

        Returns:
            torch.Tensor: The predicted action.

        """
        observation = observation.squeeze()
        actions = self.qtable[tuple(observation)]
        actions_idx = torch.where(actions == actions.max())[0]
        return actions_idx[torch.randint(0, len(actions_idx), size=(1,))]


def confidence_band(data):
    mean = np.mean(data)
    std = np.std(data)
    n = len(data)
    z = 1.96  # for a 95% confidence level
    margin_of_error = z * (std / np.sqrt(n))
    low = mean - margin_of_error
    high = mean + margin_of_error
    return mean, low, high


class QLearning(BaseAlgorithm):
    policy: QLearningPolicy

    def __init__(self, env):
        BaseAlgorithm.__init__(self, QLearningPolicy, env, 0, )
        self._setup_model()

    def _update(self, observation, action, next_observation, reward, terminated, **kwargs):
        """
        A Q-Learning algorithm implementation.

        Attributes:
            policy (QLearningPolicy): The policy used by the algorithm to select actions.

        """
        observation = observation.squeeze()
        next_observation = next_observation.squeeze()
        if "learning_rate" in kwargs and "discount_factor" in kwargs:
            learning_rate = kwargs["learning_rate"]
            discount_factor = kwargs["discount_factor"]

            idx = tuple([*observation, action.item()])

            update_term = reward if terminated else \
                reward + discount_factor * self.policy.qtable[tuple(next_observation)].max() - \
                self.policy.qtable[idx]

            self.policy.qtable[idx] = self.policy.qtable[idx] + learning_rate * update_term
        else:
            raise RuntimeError("kwargs should contain learning_rate and discount_factor")

    def learn(self: SelfBaseAlgorithm, total_timesteps: int, callback: MaybeCallback = None, log_interval: int = 100,
              tb_log_name: str = "run", reset_num_timesteps: bool = True,
              progress_bar: bool = False, log_name="", **kwargs) -> SelfBaseAlgorithm:
        """
        Train the reinforcement learning algorithm.

        Parameters:
        - self: an instance of the algorithm class, usually `self` in an object method call
        - total_timesteps (int): Total number of timesteps for training
        - callback (MaybeCallback, optional): Callback function to perform additional processing during each iteration.
          Default is None.
        - log_interval (int, optional): Log interval for TensorBoard scalar summary. Default is 100.
        - tb_log_name (str, optional): Name for the TensorBoard scalar summary run. Default is "run".
        - reset_num_timesteps (bool, optional): Whether to reset the number of timesteps in the algorithm after each run.
          Default is True.
        - progress_bar (bool, optional): Whether to display a progress bar during training. Default is False.
        - log_name (str, optional): Path for TensorBoard log directory. Default is empty string.
        - **kwargs: Additional keyword arguments for passing to the algorithm's `_update` method.

        Returns:
        - self: the instance of the algorithm class after training

        """
        writer = SummaryWriter(log_name)
        writer.add_custom_scalars_marginchart(["r", "rlow", "rhigh"])

        epsilon_start = 1.0
        epsilon_end = 0.1
        observation = self.env.reset()
        for step in tqdm(range(total_timesteps)):
            epsilon = epsilon_start - (epsilon_start - epsilon_end) * (step / total_timesteps)
            if random.random() < epsilon:
                action = torch.tensor(self.env.action_space.sample(), requires_grad=False)
            else:
                action, _ = self.policy.predict(observation)
            next_observation, reward, terminated, info = env.step(action.squeeze())
            self._update(observation, action, next_observation, reward, terminated, **kwargs)
            observation = next_observation
            if terminated:
                observation = self.env.reset()
            if step % 100 == 0:
                value, lower, upper = confidence_band(evaluate(self, env, 100))
                writer.add_scalar("r", value, step)
                writer.add_scalar("rlow", lower, step)
                writer.add_scalar("rhigh", upper, step)

        return self

    def _setup_model(self) -> None:
        self.policy = QLearningPolicy(env.observation_space, env.action_space)


class SymbolicQLearning(QLearning):
    """
    SymbolicQLearning class inherits from QLearning class and extends its functionality by adding a distance function.

    Parameters:
    env (gym.Env): The environment for the algorithm to interact with.
    distance_function: A function to compute the distance between two states.

    Attributes:
    distance_function: A function to compute the distance between two states.
    """
    def __init__(self, env: gym.Env, distance_function):
        QLearning.__init__(self, env)
        self.distance_function = distance_function

    @lru_cache(maxsize=None)
    def _get_weights(self, observation, radius):
        """
        Calculate weights for a given observation based on the radius.

        This function computes the weights for a given observation in the environment using a distance function and the
        radius. The weight for a state is computed using the distance function between the state and the observation,
        and is then normalized by the radius.

        Args:
        observation (Tuple or Array-like): An observation from the environment.
        radius (float): A scalar value that controls the spread of the weights.

        Returns:
        Tensor: A tensor containing the computed weights.
        """
        def dist(x, y):
            return self.distance_function((x, y), observation)

        return torch.exp(-torch.from_numpy(
            np.fromfunction(np.vectorize(dist), get_dims(self.env.observation_space), dtype=int)
        ) / radius)

    def _update(self, observation, action, next_observation, reward, terminated, **kwargs):
        """
        This method performs the Symbolic Q-Learning update for a given observation, action, next_observation, reward,
        and termination flag.

        The update equation is defined as:
        Q(s, a) = Q(s, a) + learning_rate * weights * (reward + discount_factor * max(Q(next_observation)) - Q(s, a))

        where weights are determined by the given distance_function and radius.

        Inputs:
        observation (np.ndarray): the current state
        action (np.ndarray): the action taken
        next_observation (np.ndarray): the next state
        reward (float): the reward received
        terminated (bool): whether the episode terminated after this step
        **kwargs:
            learning_rate (float): the learning rate to be used in the update
            discount_factor (float): the discount factor to be used in the update
            radius (float): the radius for the weights calculation

        Raises:
        RuntimeError: if kwargs does not contain learning_rate, discount_factor and radius.
        """
        observation = observation.squeeze()
        next_observation = next_observation.squeeze()
        action = action.squeeze()
        if "learning_rate" in kwargs and "discount_factor" in kwargs and "radius" in kwargs:
            learning_rate = kwargs["learning_rate"]
            discount_factor = kwargs["discount_factor"]
            radius = kwargs["radius"]

            weights = learning_rate * self._get_weights(tuple(observation), radius)

            update_term = torch.tensor(reward) if terminated else \
                reward + discount_factor * self.policy.qtable[tuple(next_observation)].max() - \
                self.policy.qtable[..., action]

            self.policy.qtable[..., action] = \
                self.policy.qtable[..., action] + weights * update_term
        else:
            raise RuntimeError("kwargs should contain learning_rate, discount_factor and radius")


def evaluate(agent, env, n_roll):
    """
    Evaluate the performance of an agent in a given environment.

    Parameters:
        agent (object): An instance of a reinforcement learning agent.
        env (gym.Env): An environment from OpenAI Gym.
        n_roll (int): The number of rollouts to perform.

    Returns:
        rewards (numpy.ndarray): An array of accumulated rewards, one per rollout.
    """
    rewards = np.zeros(n_roll)
    for i in range(n_roll):
        observation = env.reset(seed=i)
        for _ in range(100):
            action, _states = agent.predict(observation)
            observation, reward, terminated, info = env.step(action)
            rewards[i] += reward
            if terminated:
                break
    return rewards


if __name__ == "__main__":

    env = SymbolicEnv()
    check_env(env)

    model = QLearning(env)
    model.learn(total_timesteps=10000, learning_rate=0.1, discount_factor=0.5, log_name="runs/ql")

    for r in np.linspace(0.01, 0.25, 10):
        model = SymbolicQLearning(env, env.dist)
        model.learn(total_timesteps=10000, learning_rate=0.1, discount_factor=0.5, radius=r, log_name=f"runs/symbolic-ql-{r}")

