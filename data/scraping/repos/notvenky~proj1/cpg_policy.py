from stable_baselines3.common.type_aliases import Schedule
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy

from functools import partial
from typing import Callable

class CPGPolicy(nn.Module):
    def __init__(self, num_actuators, frequencies=None, amplitudes=None, phases=None, init_std=0.5):
        #repr_dim, action_shape, feature_dim, hidden_dim
        super().__init__()
        if frequencies is None:
            self.frequencies = nn.Parameter(torch.randn(num_actuators)) # softplus/exp/
        else:
            self.frequencies = nn.Parameter(frequencies) # softplus/exp/
        if amplitudes is None:
            self.amplitudes = nn.Parameter(torch.randn(num_actuators)) # softplus/exp/
        else:
            self.amplitudes = nn.Parameter(amplitudes) # softplus/exp/
        if phases is None:
            self.phases = nn.Parameter(torch.randn(num_actuators)) # softplus/exp/
        else:
            self.phases = nn.Parameter(phases) # softplus/exp/

        self.std = nn.Parameter(torch.ones(num_actuators) * init_std)

        self.num_actuators = num_actuators
        self.range = torch.tensor([np.pi/2, np.pi, np.pi/2, np.pi, np.pi/2])

    def underlying_params(self):
        return self.frequencies, self.amplitudes, self.phases

    def true_params(self):
        frequencies = F.softplus(self.frequencies)
        amplitudes = F.tanh(self.amplitudes)*self.range.to(self.amplitudes.device)
        phases = F.softplus(self.phases)
        return frequencies, amplitudes, phases
    
    def get_true_params(self):
        freq, amp, phase = self.true_params()
        return {
            'Frequency': freq.cpu().detach().numpy(),
            'Amplitude': amp.cpu().detach().numpy(),
            'Phase': phase.cpu().detach().numpy()
        }
    
    def print_true_params(self):
        freq, amp, phase = self.true_params()
        print("Frequency: ", freq)
        print("Amplitude: ", amp)
        print("Phase: ", phase)

    def log_params(self, writer, step):
        """
        Log parameters to tensorboard.

        :param writer: Tensorboard SummaryWriter instance.
        :param step: The current training step.
        """
        freq, amp, phase = self.true_params()
        
        # Log parameters to tensorboard
        writer.add_histogram('Frequency', freq, step)
        writer.add_histogram('Amplitude', amp, step)
        writer.add_histogram('Phase', phase, step)

    def forward(self, obs):
        b = obs.shape[0]
        t = obs[..., -1]
        mu = torch.zeros(b, self.num_actuators, device=t.device)
        f, a, p = self.true_params()
        # Apply oscillation
        for i in range(mu.shape[-1]):
            mu[:, i] += a[i] * torch.sin(2 * np.pi * f[i] * t + p[i])
        return mu

class PassThroughObs(nn.Module):
    def __init__(self, d_in, last_layer_dim_pi=5, last_layer_dim_vf=64+5) -> None:
        super().__init__()
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # self.policy_net = nn.Sequential(
        #     nn.Linear(d_in, last_layer_dim_vf),
        #     nn.ReLU()
        # )
        self.value_net = nn.Sequential(
            nn.Linear(d_in, last_layer_dim_vf), nn.ReLU()
        )

    def forward(self, x):
        return self.forward_actor(x), self.forward_critic(x)

    def forward_actor(self, features):
        return features

    def forward_critic(self, features):
        return self.value_net(features)

def make_cpg_policy(make_act_fn):
    class SB3CPGPolicy(ActorCriticPolicy):
        def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            lr_schedule: Callable[[float], float] = lambda x: 1e-4,
            *args,
            **kwargs,
        ):
            # Disable orthogonal initialization
            kwargs["ortho_init"] = False
            super().__init__(
                observation_space,
                action_space,
                lr_schedule,
                # Pass remaining arguments to base class
                *args,
                **kwargs,
            )

        def _build_mlp_extractor(self) -> None:
            self.mlp_extractor = PassThroughObs(self.features_dim)


        # def _build_mlp_extractor(self) -> None:
        #     na = self.action_space.shape[0]
        #     self.mlp_extractor = MyActor(na)
        def _build(self, lr_schedule: Schedule) -> None:
            """
            Create the networks and the optimizer.

            :param lr_schedule: Learning rate schedule
                lr_schedule(1) is the initial learning rate
            """
            self._build_mlp_extractor()

            latent_dim_pi = self.mlp_extractor.latent_dim_pi

            # if isinstance(self.action_dist, DiagGaussianDistribution):
            #     self.action_net, self.log_std = self.action_dist.proba_distribution_net(
            #         latent_dim=latent_dim_pi, log_std_init=self.log_std_init
            #     )
            # elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            #     self.action_net, self.log_std = self.action_dist.proba_distribution_net(
            #         latent_dim=latent_dim_pi, latent_sde_dim=latent_dim_pi, log_std_init=self.log_std_init
            #     )
            # elif isinstance(self.action_dist, (CategoricalDistribution, MultiCategoricalDistribution, BernoulliDistribution)):
            #     self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
            # else:
            #     raise NotImplementedError(f"Unsupported distribution '{self.action_dist}'.")
            d_a = self.action_space.shape[0]
            d_obs = self.observation_space.shape[0]
            # self.action_net = MyActor(d_a)
            self.action_net = make_act_fn(d_obs, d_a)
            self.log_std = nn.Parameter(torch.ones(d_a) * self.log_std_init, requires_grad=True)
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
                if not self.share_features_extractor:
                    # Note(antonin): this is to keep SB3 results
                    # consistent, see GH#1148
                    del module_gains[self.features_extractor]
                    module_gains[self.pi_features_extractor] = np.sqrt(2)
                    module_gains[self.vf_features_extractor] = np.sqrt(2)

                for module, gain in module_gains.items():
                    module.apply(partial(self.init_weights, gain=gain))

            # Setup optimizer with initial learning rate
            self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    return SB3CPGPolicy

if __name__ == '__main__':
    from stable_baselines3 import PPO
    from wriggly_train.envs.wriggly.robots import wriggly_from_swimmer
    import dmc2gymnasium
    env = dmc2gymnasium.DMCGym("wriggly", "move")
    trainer = PPO(CPGPolicy, env)
    trainer.learn(1000, progress_bar=True)