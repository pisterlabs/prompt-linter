import torch as th
import torch.nn as nn
from functools import partial
from typing import Optional, Tuple, Type
import numpy as np
import gym
from stable_baselines3.common.preprocessing import preprocess_obs
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, NatureCNN


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        lr_schedule_vf: Optional[Schedule] = None,
        ortho_init: bool = True,
        features_extractor_class: Type[BaseFeaturesExtractor] = NatureCNN,
        shared_features_extractor: bool = False,
    ):
        self.shared_features_extractor = shared_features_extractor
        self.lr_vf = lr_schedule_vf(1) if lr_schedule_vf else None

        super(CustomActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=None,
            activation_fn=nn.Tanh,
            ortho_init=ortho_init,
            use_sde=False,
            log_std_init=0.0,
            full_std=True,
            sde_net_arch=None,
            use_expln=False,
            squash_output=False,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=None,
            normalize_images=True,
            optimizer_class=th.optim.Adam,
            optimizer_kwargs=None,
        )

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """

        self.action_net = self.action_dist.proba_distribution_net(
            latent_dim=self.features_extractor.features_dim
        )

        if not self.shared_features_extractor:
            self.features_extractor_vf = self.features_extractor_class(
                self.observation_space
            )
            self.value_net = nn.Linear(self.features_extractor_vf.features_dim, 1)
            self.advantage_net = nn.Linear(
                self.features_extractor_vf.features_dim, self.action_space.n
            )
        else:
            self.value_net = nn.Linear(self.features_extractor.features_dim, 1)
            self.advantage_net = nn.Linear(
                self.features_extractor.features_dim, self.action_space.n
            )

        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            # TODO: check for features_extractor
            # Values from stable-baselines.
            # features_extractor/mlp values are
            # originally from openai/baselines (default gains/init_scales).
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
                self.advantage_net: 0.1,
            }
            if not self.shared_features_extractor:
                module_gains[self.features_extractor_vf] = np.sqrt(2)
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        if (self.lr_vf is not None) and not self.shared_features_extractor:
            self.modules_pi = nn.ModuleList([self.features_extractor, self.action_net])
            self.optimizer = self.optimizer_class(
                self.modules_pi.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
            )
            self.modules_vf = nn.ModuleList(
                [self.features_extractor_vf, self.value_net, self.advantage_net]
            )
            self.optimizer_vf = self.optimizer_class(
                self.modules_vf.parameters(), lr=self.lr_vf, **self.optimizer_kwargs
            )
        else:
            self.optimizer = self.optimizer_class(
                self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
            )

    def _predict(self, obs: th.Tensor, deterministic: bool = False) -> th.Tensor:

        latent = self.extract_features(obs)
        mean_actions = self.action_net(latent)
        distribution = self.action_dist.proba_distribution(mean_actions)

        return distribution.get_actions(deterministic=deterministic)

    def extract_features_vf(self, obs: th.Tensor) -> th.Tensor:
        pobs = preprocess_obs(
            obs, self.observation_space, normalize_images=self.normalize_images
        )

        return (
            self.features_extractor(pobs)
            if self.shared_features_extractor
            else self.features_extractor_vf(pobs)
        )

    def forward(
        self, obs: th.Tensor, deterministic: bool = False
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """

        latent = self.extract_features(obs)
        latent_vf = (
            latent if self.shared_features_extractor else self.extract_features_vf(obs)
        )
        mean_actions = self.action_net(latent)
        distribution = self.action_dist.proba_distribution(mean_actions)
        actions = distribution.get_actions(deterministic=deterministic)
        policies = distribution.distribution.probs
        log_policies = distribution.distribution.logits
        values = self.value_net(latent_vf)

        return actions, policies, log_policies, values

    def evaluate_actions(
        self, obs: th.Tensor, actions: th.Tensor, policies: th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:

        latent = self.extract_features(obs)
        latent_vf = (
            latent if self.shared_features_extractor else self.extract_features_vf(obs)
        )

        distribution = self._get_action_dist_from_latent(latent)
        log_probs = distribution.log_prob(actions)

        advantages_raw = self.advantage_net(latent_vf)
        advantages = advantages_raw.gather(dim=1, index=actions.unsqueeze(1)) - th.sum(
            policies * advantages_raw, dim=1, keepdim=True
        )
        values = self.value_net(latent_vf)
        return values, advantages, log_probs, distribution.entropy()

    def evaluate_state(
        self, obs: th.Tensor, policies: th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:

        latent = self.extract_features(obs)
        latent_vf = (
            latent if self.shared_features_extractor else self.extract_features_vf(obs)
        )
        distribution = self._get_action_dist_from_latent(latent)
        _policies = distribution.distribution.probs
        log_policies = distribution.distribution.logits

        advantages_raw = self.advantage_net(latent_vf)
        advantages = advantages_raw - th.sum(
            policies * advantages_raw, dim=1, keepdim=True
        )
        values = self.value_net(latent_vf)
        return values, advantages, _policies, log_policies, distribution.entropy()

    def predict_policy(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:

        latent = self.extract_features(obs)

        distribution = self._get_action_dist_from_latent(latent)
        policies = distribution.distribution.probs
        log_policies = distribution.distribution.logits

        return policies, log_policies, distribution.entropy()

    def predict_value(
        self, obs: th.Tensor, policies: Optional[th.Tensor] = None
    ) -> Tuple[th.Tensor, th.Tensor]:
        latent = self.extract_features_vf(obs)

        advantages_raw = self.advantage_net(latent)
        if policies is None:
            advantages = advantages_raw
        else:
            advantages = advantages_raw - th.sum(
                policies * advantages_raw, dim=1, keepdim=True
            )
        values = self.value_net(latent)

        return values, advantages
