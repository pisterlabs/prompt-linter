from functools import partial
from typing import Callable, Optional, Tuple, Type, Union

import gym.spaces
import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import Schedule
from torch import nn


class EquivariantMLP(nn.Module):
    """
    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim: int,
    ):
        super(EquivariantMLP, self).__init__()

        self.feature_dim = feature_dim

        # Policy network
        self.policy_net = nn.Sequential(nn.Linear(self.feature_dim, 16), nn.ELU(), nn.Linear(16, 1))
        # Value network
        self.value_net = nn.Sequential(nn.Linear(self.feature_dim, 16), nn.ELU(), nn.Linear(16, 1))

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        logits = self.policy_net(features)
        return torch.squeeze(logits)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        action_values = torch.squeeze(self.value_net(features))
        with torch.no_grad():
            action_probas = F.softmax(self.forward_actor(features), dim=-1)
        return torch.mean(action_values * action_probas, dim=-1)


class EquivariantActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        net_arch: Optional[list[Union[int, dict[str, list[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        *args,
        **kwargs,
    ):
        super(EquivariantActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            *args,
            **kwargs,
        )
        # Disable orthogonal initialization
        self.ortho_init = False

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        self._build_mlp_extractor()

        self.action_net = nn.Identity()
        self.value_net = nn.Identity()

        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            # TODO: check for features_extractor
            # Values from stable-baselines.
            # features_extractor/mlp values are
            # originally from openai/baselines (default gains/init_scales).
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
            }
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

        # self.log_std = nn.Parameter(torch.ones(self.action_dist.action_dim) * self.log_std_init, requires_grad=True)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = EquivariantMLP(self.features_dim)


class IdentityFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int):
        super(IdentityFeatureExtractor, self).__init__(observation_space, features_dim)
        self.identity = nn.Identity()

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.identity(observations)
