import logging
import numpy as np
import numpy.typing as npt
import torch
from typing import Dict

from agent.ac import combine_shape, discounted_cumsum

logger = logging.getLogger("ppo-buffer")


class PPOBuffer:
    """
    Stores trajectories for an agent interacting with its environment (using GAE),
    and provides methods for retrieving interactions from buffer (e.g. for learning).

    Inspired from OpenAI's implementation in spinning-up.
    """

    def __init__(
        self,
        state_space: npt.NDArray[np.float32],
        action_space: int,
        size: int,
        gamma: float,
        _lambda: float,
    ) -> None:

        self.gamma = gamma
        self._lambda = _lambda
        self.size = size

        # ~~~ Initialise buffers ~~~
        self.observations = np.zeros(combine_shape(size, state_space), dtype=np.float32)
        self.actions = np.zeros(size, dtype=np.float32)
        self.rewards = np.zeros(size, dtype=np.float32)
        self.rewards_to_go = np.zeros(size, dtype=np.float32)
        self.advantages = np.zeros(size, dtype=np.float32)
        self.est_values = np.zeros(size, dtype=np.float32)
        self.logp_actions = np.zeros(size, dtype=np.float32)

        # ~~~ Initialise pointers ~~~
        self.pointer = 0  # points to next free slot in buffer
        self.path_start_pointer = 0  # points to start of current path

    def store(self, obs, act, rew, val, logp) -> None:
        """Appends one timestep of agent-env. interaction to the buffer"""
        assert self.pointer < self.size
        self.observations[self.pointer] = obs
        self.actions[self.pointer] = act
        self.rewards[self.pointer] = rew
        self.est_values[self.pointer] = val
        self.logp_actions[self.pointer] = logp
        # ^ rewards-to-go and advantages computed & stored after path finishes

        # Increment pointer
        self.pointer += 1

    def finish_path(self, last_value: float = 0.0) -> None:
        """
        To be called at the end of a trajectory, or when a trajectory is terminated early
        (e.g. when an epoch ends, or during timeout).

        Computes rewards-to-go and (normalised) advantages for the trajectory (using lambda-GAE).
        When a trajectory is cut off early, uses the estimated value function for the last state,
        `last_value` V(s_T), to bootstrap the rewards-to-go calculation.
        """
        path_slice = slice(self.path_start_pointer, self.pointer)
        # Retrieve rewards and est. values for current path, then add `last_value` as bootstrap
        # (to avoid underestimation if path terminates early). Note `last_value = 0` when the
        # path terminates naturally, so we don't introduce bias there.
        path_rewards = np.append(self.rewards[path_slice], last_value)
        path_values = np.append(self.est_values[path_slice], last_value)

        # Calculate discounted rewards-to-go
        path_rewards_to_go = discounted_cumsum(path_rewards, self.gamma)[:-1]
        self.rewards_to_go[path_slice] = path_rewards_to_go
        # Calculate lambda-GAE
        path_deltas = (
            path_rewards[:-1] + self.gamma * path_values[1:] - path_values[:-1]
        )
        path_advantages = discounted_cumsum(path_deltas, self.gamma * self._lambda)
        self.advantages[path_slice] = path_advantages

        # Update index for start of next path
        self.path_start_pointer = self.pointer

    def get(self) -> Dict[str, torch.Tensor]:
        """
        Returns a batch of agent-env. interactions from the buffer (for learning) as torch.Tensors
        & resets pointers for next trajectory.

        Returns: a dict with:
            observations
            actions
            returns, i.e. rewards-to-go
            advantages
            action log-probs, i.e. log[pi(a_t|s_t)]
            predicted values, i.e. V_{pred}(s_t)
        """
        assert self.pointer == self.size  # buffer must be full

        # ~~ Prepare data for output ~~
        # Normalise advantages
        self.advantages = (self.advantages - self.advantages.mean()) / (
            self.advantages.std() + 1e-7
        )
        data = {
            "obs": self.observations,
            "act": self.actions,
            "ret": self.rewards_to_go,
            "adv": self.advantages,
            "logp_a": self.logp_actions,
            "vf": self.est_values,
        }
        # Convert to tensors
        data = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}

        # Reset pointers
        self.pointer = 0
        self.path_start_pointer = 0

        return data
