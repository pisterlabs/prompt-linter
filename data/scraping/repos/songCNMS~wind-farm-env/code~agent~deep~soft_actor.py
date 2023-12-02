from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

LOG_STD_MAX = 2
LOG_STD_MIN = -20


class SoftActor(nn.Module):

    def __init__(self,
                 state_shape,
                 action_shape,
                 action_scale,
                 action_bias,
                 hidden_sizes: Union[int, Tuple[int]] = (128, 128),
                 device=None):
        super().__init__()
        self._device = device
        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]
        hidden_sizes = list(hidden_sizes)
        hidden_sizes.insert(0, state_shape)
        if len(hidden_sizes) > 1:
            self._fcs = [
                nn.Linear(hidden_sizes[i-1], hidden_sizes[i], device=self._device)
                for i in range(1, len(hidden_sizes))
            ]
        else:
            self._fcs = []
        self._fc_mu = nn.Linear(hidden_sizes[-1], action_shape, device=self._device)
        self._fc_log_std = nn.Linear(hidden_sizes[-1], action_shape, device=self._device)
        self._action_scale = action_scale
        self._action_bias = action_bias

    def forward(self, x, deterministic=False, with_logprob=True):
        for fc in self._fcs:
            x = F.relu(fc(x))

        mu = self._fc_mu(x)
        log_std = self._fc_log_std(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # From OpenAI's SpinningUp
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(axis=1)
        else:
            logp_pi = None
        pi_action = torch.tanh(pi_action)
        pi_action = pi_action * self._action_scale + self._action_bias
        return pi_action, logp_pi
