import collections
import warnings
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym
import numpy as np
import torch as th
from torch import nn

from trajectory_bc_baselines.multimodal.MultimodalDiagGaussianDistribution import MultimodalDiagGaussianDistribution
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
    MlpExtractor,
    NatureCNN,
)
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.policies import BasePolicy


class MultimodalActorCriticPolicy(BasePolicy):
    """
    Policy class for actor-critic algorithms (has both policy and value prediction).
    Used by A2C, PPO and the likes.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param ortho_init: Whether to use or not orthogonal initialization
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE
    :param sde_net_arch: Network architecture for extracting features
        when using gSDE. If None, the latent features from the policy will be used.
        Pass an empty list to use the states as features.
    :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param squash_output: Whether to squash the output using a tanh function,
        this allows to ensure boundaries when using gSDE.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        num_components: int,
        entropy_approximation_mode: str,
        train_categorical_weights: bool,
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):

        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            # Small values to avoid NaN in Adam optimizer
            if optimizer_class == th.optim.Adam:
                optimizer_kwargs["eps"] = 1e-5

        super(MultimodalActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=squash_output,
        )

        # Default network architecture, from stable-baselines
        if net_arch is None:
            if features_extractor_class == NatureCNN:
                net_arch = []
            else:
                net_arch = [dict(pi=[64, 64], vf=[64, 64])]

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.ortho_init = ortho_init

        self.features_extractor = features_extractor_class(self.observation_space, **self.features_extractor_kwargs)
        self.features_dim = self.features_extractor.features_dim

        self.normalize_images = normalize_images
        self.log_std_init = log_std_init
        dist_kwargs = None
        # Keyword arguments for gSDE distribution
        if use_sde:
            dist_kwargs = {
                "full_std": full_std,
                "squash_output": squash_output,
                "use_expln": use_expln,
                "learn_features": False,
            }

        if sde_net_arch is not None:
            warnings.warn("sde_net_arch is deprecated and will be removed in SB3 v2.4.0.", DeprecationWarning)

        self.use_sde = use_sde
        self.dist_kwargs = dist_kwargs

        self.num_components = num_components

        # Action distribution
        # self.action_dist = make_proba_distribution(action_space, use_sde=use_sde, dist_kwargs=dist_kwargs)
        self.action_dist = MultimodalDiagGaussianDistribution(num_components=self.num_components,
                                                              action_dim=get_action_dim(action_space),
                                                              entropy_approximation_mode=entropy_approximation_mode,
                                                              train_categorical_weights=train_categorical_weights)
        self._build(lr_schedule)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        default_none_kwargs = self.dist_kwargs or collections.defaultdict(lambda: None)

        data.update(
            dict(
                net_arch=self.net_arch,
                activation_fn=self.activation_fn,
                use_sde=self.use_sde,
                log_std_init=self.log_std_init,
                squash_output=default_none_kwargs["squash_output"],
                full_std=default_none_kwargs["full_std"],
                use_expln=default_none_kwargs["use_expln"],
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                ortho_init=self.ortho_init,
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
            )
        )
        return data

    def _build_mlp_extractor(self) -> None:
        """
        Create the policy and value networks.
        Part of the layers can be shared.
        """
        # Note: If net_arch is None and some features extractor is used,
        #       net_arch here is an empty list and mlp_extractor does not
        #       really contain any layers (acts like an identity module).
        self.mlp_extractor = MlpExtractor(
            self.features_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
        )

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        self._build_mlp_extractor()

        latent_dim_pi = self.mlp_extractor.latent_dim_pi

        self.action_net, self.log_std = self.action_dist.proba_distribution_net(
            num_components=self.num_components,
            latent_dim=latent_dim_pi, log_std_init=self.log_std_init
        )

        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
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

    def forward(self, obs: th.Tensor, deterministic: bool = False,
                component_selection: Optional[Union[str, int]] = None) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution: MultimodalDiagGaussianDistribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic, component_selection=component_selection)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> MultimodalDiagGaussianDistribution:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        weights, mean_actions = self.action_net(latent_pi)
        return self.action_dist.proba_distribution(weights, mean_actions, self.log_std)

    def _predict(self, observation: th.Tensor, deterministic: Tuple[bool, Optional[Union[int, str]]] = (False, None)) -> th.Tensor:
        """
        Get the action according to the policy for a given observation.
        For Mixture models, we have different options to select a component to predict with. We merge this into the
        "deterministic"

        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """
        if isinstance(deterministic, bool):
            component_selection = None
        else:
            assert len(deterministic) == 2, "Need to provide (deterministic, component_selection) " \
                                            f"in deterministic parameter, given '{deterministic}'"
            deterministic, component_selection = deterministic
        return self.get_distribution(observation).get_actions(deterministic=deterministic,
                                                              component_selection=component_selection)

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs:
        :param actions:
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        return values, log_prob, distribution.entropy()

    def get_distribution(self, obs: th.Tensor) -> MultimodalDiagGaussianDistribution:
        """
        Get the current policy distribution given the observations.

        :param obs:
        :return: the action distribution.
        """
        features = self.extract_features(obs)
        latent_pi = self.mlp_extractor.forward_actor(features)
        return self._get_action_dist_from_latent(latent_pi)

    def predict_values(self, obs: th.Tensor) -> th.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs:
        :return: the estimated values.
        """
        features = self.extract_features(obs)
        latent_vf = self.mlp_extractor.forward_critic(features)
        return self.value_net(latent_vf)
