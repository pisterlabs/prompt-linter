import gym
import numpy as np
import torch as th
from torch import nn
from functools import partial
from torch.distributions import Normal, TanhTransform, TransformedDistribution
from torch.nn.functional import softplus
from typing import Any, Dict, Optional, Tuple, Type, Union

from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import FlattenExtractor
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.distributions import Distribution, sum_independent_dims


class SquashedGaussianDistribution(Distribution):
    """
    Gaussian distribution clamped with a Tanh transformation to keep samples
    between the range -1 and 1.

    :param action_dim:  Dimension of the action space.
    """

    def __init__(self, action_dim: int, use_state_dependent_std: bool = False):
        super().__init__()
        self.action_dim = action_dim
        self.mean_actions = None
        self.log_std = None
        self.use_state_dependent_std = use_state_dependent_std

    def proba_distribution_net(self, latent_dim: int, log_std_init: float = 0.0) -> Tuple[nn.Module, Union[nn.Parameter, nn.Module]]:
        """
        Create the layers and parameter that represent the distribution:
        one output will be the mean of the Gaussian, the other parameter will be the
        standard deviation (log std in fact to allow negative values)

        :param latent_dim: Dimension of the last layer of the policy (before the action layer)
        :param log_std_init: Initial value for the log standard deviation
        :return:
        """
        std_init = np.exp(log_std_init)
        std_param_init = np.log(np.exp(std_init) - 1)

        mean_actions = nn.Linear(latent_dim, self.action_dim)

        if self.use_state_dependent_std:
            std_model = nn.Linear(latent_dim, self.action_dim)
            nn.init.orthogonal_(std_model.weight, gain=0.01)
            std_model.bias.data.fill_(std_param_init)
        else:
            std_model = nn.Parameter(th.ones(self.action_dim) * std_param_init, requires_grad=True)

        return mean_actions, std_model

    def proba_distribution(self, mean_actions: th.Tensor, unnormalized_std: th.Tensor) -> "SquashedGaussianDistribution":
        """
        Create the distribution given its parameters (mean, std)

        :param mean_actions:
        :param unnormalized_std:
        :return:
        """
        action_std = th.ones_like(mean_actions) * softplus(unnormalized_std)
        base_dist = Normal(mean_actions, action_std)
        self.distribution = TransformedDistribution(base_dist, transforms=[TanhTransform()])
        return self

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        """
        Get the log probabilities of actions according to the distribution.
        Note that you must first call the ``proba_distribution()`` method.

        :param actions:
        :return:
        """
        log_prob = self.distribution.log_prob(actions)
        return sum_independent_dims(log_prob)

    def entropy(self) -> Optional[th.Tensor]:
        return None

    def sample(self) -> th.Tensor:
        # Reparametrization trick to pass gradients
        return self.distribution.rsample()

    def mode(self) -> th.Tensor:
        return th.tanh(self.distribution.base_dist.mean)

    def actions_from_params(
        self, mean_actions: th.Tensor, unnormalized_std: th.Tensor, deterministic: bool = False
    ) -> th.Tensor:
        # Update the proba distribution
        self.proba_distribution(mean_actions, unnormalized_std)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, mean_actions: th.Tensor, unnormalized_std: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        Compute the log probability of taking an action
        given the distribution parameters.

        :param mean_actions:
        :param unnormalized_std:
        :return:
        """
        actions = self.actions_from_params(mean_actions, unnormalized_std)
        log_prob = self.log_prob(actions)
        return actions, log_prob


class SimplePolicy(ActorCriticPolicy):

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        learning_rate: Schedule,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        log_std_init: float = -0.36,
        use_state_dependent_std: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):

        self.use_state_dependent_std = use_state_dependent_std
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=learning_rate,
            net_arch=[dict(pi=[32, 32], vf=[64, 64])],
            activation_fn=activation_fn,
            ortho_init=ortho_init,
            log_std_init=log_std_init,
            features_extractor_class=FlattenExtractor,
            features_extractor_kwargs=None,
            normalize_images=False,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        self._build_mlp_extractor()

        latent_dim_pi = self.mlp_extractor.latent_dim_pi

        self.action_dist = SquashedGaussianDistribution(
            action_dim=get_action_dim(self.action_space),
            use_state_dependent_std=self.use_state_dependent_std
        )
        self.action_net, self.std_model = self.action_dist.proba_distribution_net(
            latent_dim=latent_dim_pi,
            log_std_init=self.log_std_init
        )

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
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> Distribution:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        mean_actions = self.action_net(latent_pi)
        if self.use_state_dependent_std:
            std_unnormalized = self.std_model(latent_pi)
        else:
            std_unnormalized = self.std_model

        return self.action_dist.proba_distribution(mean_actions, std_unnormalized)

    def get_std(self, obs: th.Tensor) -> th.Tensor:
        if self.use_state_dependent_std:
            features = self.extract_features(obs)
            latent_pi, _ = self.mlp_extractor(features)
            unnormalized_std = self.std_model(latent_pi)
        else:
            unnormalized_std = self.std_model

        return softplus(unnormalized_std)

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value, log probability and mean of the action
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1,) + self.action_space.shape)
        return actions, values, log_prob, distribution.mode()
