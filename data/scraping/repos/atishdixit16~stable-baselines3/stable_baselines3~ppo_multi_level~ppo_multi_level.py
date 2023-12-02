from tkinter import N
import warnings
from typing import Any, Dict, Optional, Type, Union
from matplotlib import pyplot as plt

import numpy as np
import os
from sqlalchemy import false
import torch as th
from gym import spaces
from torch.nn import functional as F

from stable_baselines3.common.on_policy_algorithm_multi_level import OnPolicyAlgorithmMultiLevel
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.pymlmc import mlmc_ppo
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn



class PPO_ML(OnPolicyAlgorithmMultiLevel):
    """
    Proximal Policy Optimization algorithm (PPO) (clip version)

    Paper: https://arxiv.org/abs/1707.06347
    Code: This implementation borrows code from OpenAI Spinning Up (https://github.com/openai/spinningup/)
    https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail and
    and Stable Baselines (PPO2 from https://github.com/hill-a/stable-baselines)

    Introduction to PPO: https://spinningup.openai.com/en/latest/algorithms/ppo.html

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel)
        NOTE: n_steps * n_envs must be greater than 1 (because of the advantage normalization)
        See https://github.com/pytorch/pytorch/issues/29372
    :param batch_size: Minibatch size
    :param n_epochs: Number of epoch when optimizing the surrogate loss
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param clip_range: Clipping parameter, it can be a function of the current progress
        remaining (from 1 to 0).
    :param clip_range_vf: Clipping parameter for the value function,
        it can be a function of the current progress remaining (from 1 to 0).
        This is a parameter specific to the OpenAI implementation. If None is passed (default),
        no clipping will be done on the value function.
        IMPORTANT: this clipping depends on the reward scaling.
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param target_kl: Limit the KL divergence between updates,
        because the clipping is not enough to prevent large update
        see issue #213 (cf https://github.com/hill-a/stable-baselines/issues/213)
        By default, there is no limit on the kl div.
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: 'dict[int: Union[GymEnv, str]]',
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: 'dict[int: int]' = {1:2048},
        batch_size: 'dict[int: int]' = {1:64},
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        target_kl: Optional[float] = None,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):

        super(PPO_ML, self).__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )

        # Sanity check, otherwise it will lead to noisy gradient and NaN
        # because of the advantage normalization
        self.batch_size_dict = batch_size

        if env is not None:
            for level in self.env_dict.keys():
                assert (
                    self.batch_size_dict[level] > 1
                ), "`batch_size` must be greater than 1. See https://github.com/DLR-RM/stable-baselines3/issues/440"

                if self.env_dict[level] is not None:
                    # Check that `n_steps * n_envs > 1` to avoid NaN
                    # when doing advantage normalization
                    buffer_size = self.env_dict[level].num_envs * self.n_steps_dict[level]
                    assert (
                        buffer_size > 1
                    ), f"`n_steps * n_envs` must be greater than 1. Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}"
                    # Check that the rollout buffer size is a multiple of the mini-batch size
                    untruncated_batches = buffer_size // self.batch_size_dict[level]
                    if buffer_size % self.batch_size_dict[level] > 0:
                        warnings.warn(
                            f"You have specified a mini-batch size of {batch_size[len(batch_size)]},"
                            f" but because the `RolloutBuffer` is of size `n_steps * n_envs = {buffer_size}`,"
                            f" after every {untruncated_batches} untruncated mini-batches,"
                            f" there will be a truncated mini-batch of size {buffer_size % batch_size[0]}\n"
                            f"We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.\n"
                            f"Info: (n_steps={self.n_steps} and n_envs={self.env.num_envs})"
                        )
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.target_kl = target_kl

        # check multi-level variables
        if env is not None:
            self._check_multi_level_variables()

        if _init_setup_model:
            self._setup_model()

    def _check_multi_level_variables(self):

        env_keys_list = list(self.env_dict.keys())
        nstep_keys_list = list(self.n_steps_dict.keys())
        batch_size_keys_list = list(self.batch_size_dict.keys())

        assert np.array_equal(np.diff(np.sort(env_keys_list)), np.ones(len(self.env_dict)-1) ), "levels must cover for level 1 to L each representing environment grid fidelity in ascending order "

        assert np.array_equal(np.sort(env_keys_list), np.sort(nstep_keys_list) ) , "`env_dict` and `n_step_dict` must have equal number of levels"
        assert np.array_equal(np.sort(env_keys_list), np.sort(batch_size_keys_list) ) , "`env_dict` and `batch_size_dict` must have equal number of levels"

        ratio = self.n_steps_dict[1]/ self.batch_size_dict[1]
        for t,m in zip(self.n_steps_dict.values(), self.batch_size_dict.values()):
            assert t/m==ratio, "ratio of n_steps to batch_size should be equal on all levels"

        # make sure to set kl_target to None
        # assert self.target_kl==None, 'set target kl to None since the kl-based learning truncation is not considered in multi-level implementation'

    def _setup_model(self) -> None:
        super(PPO_ML, self)._setup_model()

        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

    def compute_batch_losses(self, rollout_data, clip_range, clip_range_vf):
        actions = rollout_data.actions
        if isinstance(self.action_space, spaces.Discrete):
            # Convert discrete action from float to long
            actions = rollout_data.actions.long().flatten()

        # Re-sample the noise matrix because the log_std has changed
        if self.use_sde:
            self.policy.reset_noise(self.batch_size_array[0])

        values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
        values = values.flatten()
        # Normalize advantage
        advantages = rollout_data.advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ratio between old and new policy, should be one at the first iteration
        ratio = th.exp(log_prob - rollout_data.old_log_prob)

        # clipped surrogate loss
        policy_loss_1 = advantages * ratio
        policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
        policy_batch_loss = -th.min(policy_loss_1, policy_loss_2)

        if self.clip_range_vf is None:
            # No clipping
            values_pred = values
        else:
            # Clip the different between old and new value
            # NOTE: this depends on the reward scaling
            values_pred = rollout_data.old_values + th.clamp(
                values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                )
        # Value loss using the TD(gae_lambda) target
        value_batch_loss = th.square(rollout_data.returns - values_pred)

        # Entropy loss favor exploration
        if entropy is None:
            # Approximate entropy when no analytical form
            entropy_batch_loss = -log_prob
        else:
            entropy_batch_loss = entropy

        return policy_batch_loss, value_batch_loss, entropy_batch_loss, ratio


    def update_policy(self, loss) -> None:
        # Optimization step
        self.policy.optimizer.zero_grad()
        loss.backward()
        # Clip grad norm
        th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.policy.optimizer.step()


    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)
        else:
            clip_range_vf = None

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):

            batch_generator = {}
            for level in self.rollout_buffer_dict.keys():
                batch_generator[level] = self.rollout_buffer_dict[level].get_sync(self.sync_rollout_buffer_dict[level], self.batch_size_dict[level])


            for _ in range( int(np.ceil(self.n_steps_dict[1]*self.n_envs/self.batch_size_dict[1])) ):
                
                loss_mlmc = 0.0
                approx_kl_divs = []
                # Do a complete pass on the rollout buffer
                # for rollout_data in self.rollout_buffer_array[0].get(self.batch_size_array[0]):
                for level in self.rollout_buffer_dict.keys():

                    # compute batch loss on rollout buffer
                    policy_batch_loss, value_batch_loss, entropy_batch_loss, ratio = self.compute_batch_losses(next(batch_generator[level]), clip_range, clip_range_vf)
                    
                    # compute batch loss on sync rollout buffer
                    if level > 1:
                        policy_batch_loss_, value_batch_loss_, entropy_batch_loss_, _ = self.compute_batch_losses(next(batch_generator[level]), clip_range, clip_range_vf)
                    else:
                        _ = next(batch_generator[level]) # next run to rollout sync buffer at level
                        policy_batch_loss_, value_batch_loss_, entropy_batch_loss_ = 0,0,0

                    # averaging loss difference terms of MLMC estimator at current level
                    policy_loss = th.mean(policy_batch_loss - policy_batch_loss_)
                    value_loss = th.mean(value_batch_loss - value_batch_loss_)
                    entropy_loss = -th.mean(entropy_batch_loss - entropy_batch_loss_)

                    # Logging
                    pg_losses.append(policy_batch_loss.mean().item())
                    value_losses.append(value_batch_loss.mean().item())
                    entropy_losses.append(entropy_batch_loss.mean().item())
                    batch_clip_fraction = (th.abs(ratio - 1) > clip_range).float()
                    clip_fraction = th.mean(batch_clip_fraction).item()
                    clip_fractions.append(clip_fraction)


                    # Calculate a}p}proximate form of reverse KL Divergence for early stopping
                    # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                    # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                    # and Schulman blog: http://joschu.net/blog/kl-approx.html
                    with th.no_grad():
                        log_ratio = th.log(ratio)
                        approx_kl_div = th.mean((ratio - 1) - log_ratio).cpu().numpy()
                        approx_kl_divs.append(approx_kl_div)

                    if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                        continue_training = False
                        # break
                    loss_l = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
                    loss_mlmc += loss_l
                
                self.update_policy(loss_mlmc)

            if not continue_training:
                if self.verbose >= 1:
                    print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                break

        self._n_updates += self.n_epochs
        values_array = np.hstack( tuple( self.rollout_buffer_dict[level].values.flatten() for level in self.rollout_buffer_dict.keys() ) )
        returns_array = np.hstack( tuple( self.rollout_buffer_dict[level].returns.flatten() for level in self.rollout_buffer_dict.keys() ) )
        explained_var = explained_variance( values_array, returns_array )

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss_mlmc.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)


    def train_with_fine_level(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)
        else:
            clip_range_vf = None

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True

        fine_level = len(self.n_steps_dict)

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            # for rollout_data in self.rollout_buffer_array[0].get(self.batch_size_array[0]):
            for rollout_data in self.analysis_rollout_buffer_dict[fine_level].get_analysis_batch( self.batch_size_dict[fine_level] ):
                policy_batch_loss, value_batch_loss, entropy_batch_loss, ratio = self.compute_batch_losses(rollout_data, clip_range, clip_range_vf)

                # Losses 
                policy_loss = th.mean(policy_batch_loss)
                value_loss = th.mean(value_batch_loss)
                entropy_loss = -th.mean(entropy_batch_loss)

                # Logging
                pg_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
                batch_clip_fraction = (th.abs(ratio - 1) > clip_range).float()
                clip_fraction = th.mean(batch_clip_fraction).item()
                clip_fractions.append(clip_fraction)
    
                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = th.log(ratio)
                    approx_kl_div = th.mean((ratio - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                self.update_policy(loss)

            if not continue_training:
                break

        self._n_updates += self.n_epochs
        explained_var = explained_variance(self.analysis_rollout_buffer_dict[fine_level].values.flatten(), self.analysis_rollout_buffer_dict[fine_level].returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

    def analysis(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)
        else:
            clip_range_vf = None

        # compute losses and computational time at each level
        loss_dict = {}
        comp_time = {}
        for level in self.env_dict.keys():
            for rollout in self.analysis_rollout_buffer_dict[level].get_analysis_batch(None):
                with th.no_grad():
                    policy_loss, value_loss, entropy_loss, _ = self.compute_batch_losses(rollout, clip_range, clip_range_vf)
                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
                loss_dict[level] = loss.cpu().detach().numpy()
                comp_time[level] = rollout.times.cpu().detach().numpy()

        fine_level = len(self.env_dict.keys())

        # define mlmc_fn for pymlmc analysis
        def mlmc_fn(l,N):
            '''
            mlmc_fn: the user low-level routine for level l estimator. Its interface is

            (sums, cost) = mlmc_fn(l, N, *args, **kwargs)

            Inputs:  l: level
                     N: number of samples
                     *args, **kwargs: optional additional user variables

            Outputs: sums[0]: sum(Y)
                     sums[1]: sum(Y**2)
                     sums[2]: sum(Y**3)
                     sums[3]: sum(Y**4)
                     sums[4]: sum(P_l)
                     sums[5]: sum(P_l**2)
                     where Y are iid samples with expected value
                          E[P_0]            on level 0
                          E[P_l - P_{l-1}]  on level l > 0
                     cost: user-defined computational cost of N samples
            '''

            fine_level = len(self.env_dict.keys())
            assert N <= loss_dict[fine_level].shape[0], f'number of samples `N`({N}) should be smaller than `n_expt`, try increasing `n_expt`({loss_dict[fine_level].shape[0]})'
            level=l+1

            indices = np.random.choice(loss_dict[fine_level].shape[0],N, replace=False)
            p_l = loss_dict[level][indices]
            y = p_l
            cost = comp_time[level][indices]
            if level>1:
                p_l_back = loss_dict[level-1][indices]
                y = y - p_l_back
                cost = cost + comp_time[level-1][indices]
            
            sums = [0]*6
            sums[0]= sum(y)
            sums[1]= sum(y**2)
            sums[2]= sum(y**3)
            sums[3]= sum(y**4)
            sums[4]= sum(p_l)
            sums[5]= sum(p_l**2)
            cost= sum(cost)

            return np.array(sums), cost

        expt_results, mc_results, ml_results = mlmc_ppo(mlmc_fn, self.num_expt, fine_level-1, self.eps_array)

        return mc_results, ml_results, expt_results
            

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "PPO",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> "PPO_ML":

        return super(PPO_ML, self).learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            tb_log_name=tb_log_name,
            eval_log_path=eval_log_path,
            reset_num_timesteps=reset_num_timesteps,
        )

    def mlmc_analysis(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "OnPolicyAlgorithmMultiLevel",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
        n_expt: int = 100,
        eps_array: 'list[float]' = [0.1, 0.05],
        analysis_interval: int = 100,
        step_comp_time_dict: 'dict[int: float]'=None
    ):

        return super(PPO_ML, self).mlmc_analysis(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            tb_log_name=tb_log_name,
            eval_log_path=eval_log_path,
            reset_num_timesteps=reset_num_timesteps,
            n_expt=n_expt,
            eps_array=eps_array,
            analysis_interval=analysis_interval,
            step_comp_time_dict=step_comp_time_dict
        )