from functools import partial

import numpy as np
from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
)
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule
from torch import nn


class CustomActorCriticPolicy(ActorCriticPolicy):
    def get_parameter_groups(self):
        return {
            "policy": [
                *self.mlp_extractor.policy_net.parameters(),
                *self.action_net.parameters(),
            ],
            "value": [
                *self.mlp_extractor.value_net.parameters(),
                *self.value_net.parameters(),
            ],
            "log_std": [self.log_std],
        }

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        self._build_mlp_extractor()

        latent_dim_pi = self.mlp_extractor.latent_dim_pi

        if isinstance(self.action_dist, DiagGaussianDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, log_std_init=self.log_std_init
            )
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, latent_sde_dim=latent_dim_pi, log_std_init=self.log_std_init
            )
        elif isinstance(
            self.action_dist, (CategoricalDistribution, MultiCategoricalDistribution, BernoulliDistribution)
        ):
            self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
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
        #* This is where novel code has been written!
        params = self.get_parameter_groups()
        params = [dict(params=p, name=n, lr=lr_schedule[n]) for n, p in params.items()]
        num_params = sum([len(g["params"]) for g in params])
        num_params_total = len(list(self.parameters()))
        assert num_params == num_params_total, (
            f"Number of parameters should be {num_params}, " f"got {num_params_total} instead"
        )
        self.optimizer = self.optimizer_class(
            params,
            **self.optimizer_kwargs,
        )
