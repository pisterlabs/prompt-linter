import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

import numpy as np
from typing import Optional
from typing import Sequence

from soft_actor_critic.utilities import weight_initialization
from soft_actor_critic.utilities import get_multilayer_perceptron


class StochasticPolicy(nn.Module):
    def __init__(self, input_dims: int, num_actions: int, hidden_units: Optional[Sequence[int]] = None,
                 action_space=None, epsilon: float = 10e-6, log_sigma_max: float = 2, log_sigma_min: float = -20):
        super(StochasticPolicy, self).__init__()

        if hidden_units is None:
            hidden_units = [256, 256]

        self.input_dims = input_dims
        self.num_actions = num_actions
        self.hidden_units = list(hidden_units)

        self.epsilon = epsilon
        self.log_sigma_max = log_sigma_max
        self.log_sigma_min = log_sigma_min

        units = [input_dims] + list(hidden_units)
        self.multilayer_perceptron = get_multilayer_perceptron(units, keep_last_relu=True)
        self.mean_linear = nn.Linear(units[-1], num_actions)
        self.log_std_linear = nn.Linear(units[-1], num_actions)

        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor((action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor((action_space.high + action_space.low) / 2.)

        self.apply(weight_initialization)

    def forward(self, x):  # todo maybe merge forward and evaluate + maybe add the "act" from openai
        x = self.multilayer_perceptron(x)
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)

        log_std_clamped = torch.clamp(log_std, min=self.log_sigma_min, max=self.log_sigma_max)
        std = torch.exp(log_std_clamped)
        return mean, std

    def evaluate(self, state, deterministic: bool = False, with_log_probability: bool = True):
        mean, std = self.forward(state)
        distribution = Normal(mean, std)
        sample = distribution.rsample()

        if deterministic:
            action = mean
        else:
            action = torch.tanh(sample)  # todo when sac working, multiply by action_scale and add action_bias

        if with_log_probability:
            # Implementation that I originally implemented
            # the "_" are only here for now to debug the values and the shapes
            # log_probability_ = distribution.log_prob(sample) - torch.log((1 - action.pow(2)) + self.epsilon)
            # log_probability = log_probability_.sum(1, keepdim=True)

            # OPENAI Implementation
            # https://github.com/openai/spinningup/blob/038665d62d569055401d91856abb287263096178/spinup/algos/pytorch/sac/core.py#L59
            log_probability_ = distribution.log_prob(sample).sum(axis=-1, keepdim=True)
            log_probability__ = (2 * (np.log(2) - sample - F.softplus(-2 * sample))).sum(axis=1).unsqueeze(1)
            log_probability = log_probability_ - log_probability__
        else:
            log_probability = None

        return action, log_probability

    def act(self, observation, deterministic=False) -> np.array:  # todo need to replace in the agent code
        with torch.no_grad():
            action, _ = self.evaluate(observation, deterministic=deterministic, with_log_probability=False)
            return action.cpu().numpy()
