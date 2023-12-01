from typing import Any, Dict, Optional, Tuple, Type, Union

import numpy as np
from gym import spaces

import torch
import torch.nn.functional as F
from torch import distributions

from lightning_baselines3.on_policy_models.on_policy_model import OnPolicyModel
from lightning_baselines3.common.type_aliases import GymEnv
from lightning_baselines3.common.utils import explained_variance



class PPO(OnPolicyModel):
    """
    Proximal Policy Optimization algorithm (PPO) (clip version)

    Paper: https://arxiv.org/abs/1707.06347
    Code: This implementation borrows code from OpenAI Spinning Up (https://github.com/openai/spinningup/)
    https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail and
    and Stable Baselines 3 (https://github.com/DLR-RM/stable-baselines3)

    Introduction to PPO: https://spinningup.openai.com/en/latest/algorithms/ppo.html

    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param eval_env: The environment to evaluate on, must not be vectorised/parallelrised
        (if registered in Gym, can be str. Can be None for loading trained models)
    :param buffer_length: (int) Length of the buffer and the number of steps to run for each environment per update
    :param num_rollouts: Number of rollouts to do per PyTorch Lightning epoch. This does not affect any training dynamic,
        just how often we evaluate the model since evaluation happens at the end of each Lightning epoch
    :param batch_size: Minibatch size for each gradient update
    :param epochs_per_rollout: Number of epochs to optimise the loss for
    :param num_eval_episodes: The number of episodes to evaluate for at the end of a PyTorch Lightning epoch
    :param gamma: (float) Discount factor
    :param gae_lambda: (float) Factor for trade-off of bias vs variance for Generalized Advantage Estimator.
        Equivalent to classic advantage when set to 1.
    :param clip_range: Clipping parameter, it can be a function of the current progress
        remaining (from 1 to 0).
    :param clip_range_vf: Clipping parameter for the value function,
        it can be a function of the current progress remaining (from 1 to 0).
        This is a parameter specific to the OpenAI implementation. If None is passed (default),
        no clipping will be done on the value function.
        IMPORTANT: this clipping depends on the reward scaling.
    :param target_kl: Limit the KL divergence between updates,
        because the clipping is not enough to prevent large update
        see issue #213 (cf https://github.com/hill-a/stable-baselines/issues/213)
        By default, there is no limit on the kl div.
    :param value_coef: Value function coefficient for the loss calculation
    :param entropy_coef: Entropy coefficient for the loss calculation
    :param use_sde: (bool) Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration
    :param sde_sample_freq: (int) Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param verbose: The verbosity level: 0 none, 1 training information, 2 debug
    :param seed: Seed for the pseudo random generators
    """
    def __init__(
        self,
        env: Union[GymEnv, str],
        eval_env: Union[GymEnv, str],
        buffer_length: int = 2048,
        num_rollouts: int = 1,
        batch_size: int = 64,
        epochs_per_rollout: int = 10,
        num_eval_episodes: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        clip_range_vf: Optional[float] = None,
        target_kl: Optional[float] = None,
        value_coef: float = 0.5,
        entropy_coef: float = 0.0,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        verbose: int = 0,
        seed: Optional[int] = None,
    ):
        super(PPO, self).__init__(
            env=env,
            eval_env=eval_env,
            buffer_length=buffer_length,
            num_rollouts=num_rollouts,
            batch_size=batch_size,
            epochs_per_rollout=epochs_per_rollout,
            num_eval_episodes=num_eval_episodes,
            gamma=gamma,
            gae_lambda=gae_lambda,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            verbose=verbose,
            seed=seed
        )

        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.target_kl = target_kl

    def forward(
        self, x: torch.Tensor
        ) -> Tuple[distributions.Distribution, torch.Tensor]:
        """
        Runs both the actor and critic network

        :param x: The input observations
        :return: The deterministic action of the actor
        """
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        """
        Specifies the update step for PPO. Override this if you wish to modify the PPO algorithm
        """
        if self.use_sde:
            self.reset_noise(self.batch_size)

        dist, values = self(batch.observations)
        log_probs = dist.log_prob(batch.actions)
        values = values.flatten()

        advantages = batch.advantages.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        ratio = torch.exp(log_probs - batch.old_log_probs)
        policy_loss_1 = advantages * ratio
        policy_loss_2 = advantages * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
        policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

        if self.clip_range_vf:
            values = batch.old_values.detach() + torch.clamp(values - batch.old_values.detach(), -self.clip_range_vf, self.clip_range_vf)

        value_loss = F.mse_loss(batch.returns.detach(), values)

        entropy_loss = -dist.entropy().mean()
        loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

        with torch.no_grad():
            clip_fraction = torch.mean((torch.abs(ratio - 1) > self.clip_range).float())
            approx_kl = torch.mean(batch.old_log_probs - log_probs)
            explained_var = explained_variance(batch.old_values, batch.returns)
        self.log_dict({
            'train_loss': loss,
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy_loss': entropy_loss,
            'clip_fraction': clip_fraction,
            'approx_kl': approx_kl,
            'explained_var': explained_var},
            prog_bar=False, logger=True)

        return loss
