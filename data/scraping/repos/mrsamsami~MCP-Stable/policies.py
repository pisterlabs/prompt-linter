from functools import partial
from typing import Callable, Dict, List, Optional, Type, Union

import gym
import numpy as np
import torch as th
from stable_baselines3.common.distributions import (DiagGaussianDistribution,
                                                    Distribution)
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule
from torch import nn
from torch.distributions import Normal

from mcp.models import MCPPOHiddenLayers
from models import MCPHiddenLayers


class MCPNaive(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        state_dim: int = 111,
        goal_dim: int = 2,
        num_primitives: int = None,
        learn_log_std: bool = True,
        big_model: bool = True,
        *args,
        **kwargs,
    ):

        assert state_dim + goal_dim == observation_space.shape[0]
        self.mcp_state_dim = state_dim
        self.mcp_goal_dim = goal_dim
        self.num_primitives = num_primitives
        self.learn_log_std = learn_log_std
        self.big_model = big_model

        super(MCPNaive, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = MCPHiddenLayers(
            self.mcp_state_dim,
            self.mcp_goal_dim,
            int(np.prod(self.action_space.shape)),
            self.num_primitives,
            self.learn_log_std,
            self.big_model,
        )

    def freeze_primitives(self):
        self.mlp_extractor.freeze_primitives()

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        self._build_mlp_extractor()

        latent_dim_pi = self.mlp_extractor.latent_dim_pi

        if isinstance(self.action_dist, DiagGaussianDistribution):
            if self.learn_log_std:
                self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                    latent_dim=latent_dim_pi, log_std_init=self.log_std_init
                )
                self.action_net = nn.Identity()
            else:
                self.action_net = nn.Identity()
        else:
            raise NotImplementedError(f"Unsupported distribution '{self.action_dist}'.")

        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
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
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> Distribution:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        if not self.learn_log_std:
            latent_pi, latent_std = latent_pi
        assert isinstance(latent_pi, th.Tensor)
        mean_actions = self.action_net(latent_pi)

        if isinstance(self.action_dist, DiagGaussianDistribution):
            if self.learn_log_std:
                return self.action_dist.proba_distribution(mean_actions, self.log_std)
            else:
                self.action_dist.distribution = Normal(mean_actions, latent_std)
                return self.action_dist
        else:
            raise ValueError("Invalid action distribution")

    def predict_weights(self, observation: np.ndarray) -> np.ndarray:
        self.set_training_mode(False)

        observation, vectorized_env = self.obs_to_tensor(observation)

        with th.no_grad():
            features = self.extract_features(observation)
            weights = self.mlp_extractor.forward_weights(features)
        # Convert to numpy
        weights = weights.cpu().numpy()

        # Remove batch dimension if needed
        if not vectorized_env:
            weights = weights[0]

        return weights


class MPPO(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        state_dim: int = 11,
        goal_dim: int = 2,
        models: List[nn.Module] = None,
        learn_log_std: bool = False,
        big_model: bool = False,
        *args,
        **kwargs,
    ):

        assert state_dim + goal_dim == observation_space.shape[0]
        self.mcppo_state_dim = state_dim
        self.mcppo_goal_dim = goal_dim
        self.models = models
        self.learn_log_std = learn_log_std
        self.big_model = big_model

        super(MPPO, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = MCPPOHiddenLayers(
            self.mcppo_state_dim,
            self.mcppo_goal_dim,
            int(np.prod(self.action_space.shape)),
            self.models,
            self.learn_log_std,
            self.big_model,
        )

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        self._build_mlp_extractor()

        latent_dim_pi = self.mlp_extractor.latent_dim_pi

        if isinstance(self.action_dist, DiagGaussianDistribution):
            if self.learn_log_std:
                self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                    latent_dim=latent_dim_pi, log_std_init=self.log_std_init
                )
                self.action_net = nn.Identity()
            else:
                self.action_net = nn.Identity()
        else:
            raise NotImplementedError(f"Unsupported distribution '{self.action_dist}'.")

        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
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
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> Distribution:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        if not self.learn_log_std:
            latent_pi, latent_std = latent_pi
        assert isinstance(latent_pi, th.Tensor)
        mean_actions = self.action_net(latent_pi)

        if isinstance(self.action_dist, DiagGaussianDistribution):
            if self.learn_log_std:
                return self.action_dist.proba_distribution(mean_actions, self.log_std)
            else:
                self.action_dist.distribution = Normal(mean_actions, latent_std)
                return self.action_dist
        else:
            raise ValueError("Invalid action distribution")

    def predict_weights(self, observation: np.ndarray) -> np.ndarray:
        self.set_training_mode(False)

        observation, vectorized_env = self.obs_to_tensor(observation)

        with th.no_grad():
            features = self.extract_features(observation)
            weights = self.mlp_extractor.forward_weights(features)
        # Convert to numpy
        weights = weights.cpu().numpy()

        # Remove batch dimension if needed
        if not vectorized_env:
            weights = weights[0]

        return weights
