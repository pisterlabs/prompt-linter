'''
Author: Jikun Kang
Date: 2021-11-24 09:24:48
LastEditTime: 2022-10-13 17:18:11
LastEditors: Jikun Kang
FilePath: /Learning-Multi-Objective-Curricula-for-Robotic-Policy-Learning/core/custom_ppo.py
'''
from typing import Any, Callable, Dict, Optional, Type, Union

import higher
import numpy as np
import torch as th
from gym import spaces
from stable_baselines3.common import logger
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback
from stable_baselines3.common.utils import explained_variance, get_schedule_fn
from torch.nn import functional as F

from core.custom_on_policy_algorithm import OnPolicyAlgorithm
from core.custom_policies import ActorCriticPolicy
from utils.util import reverse_unroll


class PPO(OnPolicyAlgorithm):
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
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
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
        #213 (cf https://github.com/hill-a/stable-baselines/issues/213)
        see issue
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
            env: Union[GymEnv, str],
            learning_rate: Union[float, Callable] = 3e-4,
            n_steps: int = 2048,
            batch_size: Optional[int] = 64,
            n_epochs: int = 10,
            gamma: float = 0.99,
            gae_lambda: float = 0.95,
            clip_range: float = 0.2,
            clip_range_vf: Optional[float] = None,
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
            h_cell=None,
            initial_cell=None,
            reward_cell=None,
            memory_cell=None,
            shared_hypernet=None,
            tensor_log=None,
            n_inner_loops=1,
            meta=True,
            reward_shaping=False,
            initial_curriculum=False,
            goal_curriculum=False,
            all_curricula=True,
            memory_only=False,
            num_timesteps=0
    ):

        super(PPO, self).__init__(
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
            h_cell=h_cell,
            initial_cell=initial_cell,
            reward_cell=reward_cell,
            memory_cell=memory_cell,
            shared_hypernet=shared_hypernet,
            meta=meta,
            reward_shaping=reward_shaping,
            initial_curriculum=initial_curriculum,
            goal_curriculum=goal_curriculum,
            all_curricula=all_curricula,
            memory_only=memory_only,
            num_timesteps=num_timesteps
        )

        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.target_kl = target_kl
        self.h_cell = h_cell
        self.initial_cell = initial_cell
        self.reward_cell = reward_cell
        self.shared_hypernet = shared_hypernet
        self.tensor_log = tensor_log
        self.train_time = 0
        self.n_inner_loops = n_inner_loops
        self.hparams = self.h_cell.parameters()
        self.initial_hparams = self.initial_cell.parameters()
        self.initial_curriculum = initial_curriculum
        self.goal_curriculum = goal_curriculum
        self.memory_only = memory_only
        self.env = env
        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super(PPO, self)._setup_model()

        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

    def return_policy_model(self):
        return self.policy

    # def return_value_function(self):
    #     return self.

    def return_replay_buffer(self):
        return self.rollout_buffer

    def outer_loss(self, params, hparams):
        # Outer update
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(
                self._current_progress_remaining)

        for rollout_data in self.rollout_buffer.get(self.batch_size):
            actions = rollout_data.actions
            if isinstance(self.action_space, spaces.Discrete):
                # Convert discrete action from float to long
                actions = rollout_data.actions.long().flatten()

            # Re-sample the noise matrix because the log_std has changed
            # if that line is commented (as in SAC)
            if self.use_sde:
                self.policy.reset_noise(self.batch_size)

            obs_tensor = th.as_tensor(
                rollout_data.observations[:1].to(self.device))
            rule_subgoal = th.tensor(
                [[1, 0, 0]*self.env.num_envs]).view(self.env.num_envs, 3)
            rule_init = th.tensor(
                [[0, 1, 0]*self.env.num_envs]).view(self.env.num_envs, 3)
            rule_reward = th.tensor(
                [[0, 0, 1]*self.env.num_envs]).view(self.env.num_envs, 3)
            obs_tensor_subgoal = th.cat(
                (obs_tensor.float(), rule_subgoal.to(self.device)),
                dim=1)
            obs_tensor_init = th.cat(
                (obs_tensor.float(), rule_init.to(self.device)),
                dim=1)
            obs_tensor_reward = th.cat(
                (obs_tensor.float(), rule_reward.to(self.device)),
                dim=1)
            subgoals = []
            rewards = []
            init_states = []
            memories = []
            for i in range(64):
                subgoal, state, hyper_state = self.shared_hypernet(x=obs_tensor_subgoal.float(),
                                                                   state=self.state,
                                                                   hyper_state=self.hyper_state,
                                                                   lstm_cell=self.h_cell,
                                                                   emit_mem=False)
                shape_reward, state, hyper_state = self.shared_hypernet(x=obs_tensor_init.float(),
                                                                        state=self.state,
                                                                        hyper_state=self.hyper_state,
                                                                        lstm_cell=self.reward_cell,
                                                                        emit_mem=False)
                init_state, state, hyper_state = self.shared_hypernet(x=obs_tensor_reward.float(),
                                                                      state=self.state,
                                                                      hyper_state=self.hyper_state,
                                                                      lstm_cell=self.initial_cell,
                                                                      emit_mem=False)
                _, state, hyper_state, memory = self.shared_hypernet(x=obs_tensor_subgoal.float(),
                                                                     state=self.state,
                                                                     hyper_state=self.hyper_state,
                                                                     lstm_cell=self.memory_cell,
                                                                     emit_mem=True)

                subgoals.append(subgoal)
                rewards.append(shape_reward)
                init_states.append(init_state)
                memories.append(memory)

            subgoals = th.cat(subgoals, dim=0)
            rewards = th.cat(rewards, dim=0)
            init_states = th.cat(init_states, dim=0)
            memories = th.cat(memories, dim=0)

            if self.goal_curriculum:
                values, log_prob, entropy = self.policy.evaluate_actions_subgoal(rollout_data.observations,
                                                                                 subgoals,
                                                                                 actions)
            elif self.memory_only:
                values, log_prob, entropy = self.policy.evaluate_actions_memory(rollout_data.observations,
                                                                                memories,
                                                                                actions)
            elif self.reward_shaping:
                values, log_prob, entropy = self.policy.evaluate_actions_reward(rollout_data.observations,
                                                                                rewards,
                                                                                actions)
            elif self.initial_curriculum:
                values, log_prob, entropy = self.policy.evaluate_actions_init(rollout_data.observations,
                                                                              init_states,
                                                                              actions)
            elif self.all_curricula:
                values, log_prob, entropy = self.policy.evaluate_actions_all(rollout_data.observations,
                                                                             subgoals,
                                                                             rewards,
                                                                             init_states,
                                                                             memories,
                                                                             actions)

            else:
                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations,
                                                                         actions)
            values = values.flatten()
            # Normalize advantage
            advantages = rollout_data.advantages
            advantages = (advantages - advantages.mean()) / \
                (advantages.std() + 1e-8)

            # ratio between old and new policy, should be one at the first iteration
            ratio = th.exp(log_prob - rollout_data.old_log_prob)

            # clipped surrogate loss
            policy_loss_1 = advantages * ratio
            policy_loss_2 = advantages * \
                th.clamp(ratio, 1 - clip_range, 1 + clip_range)
            policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

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
            value_loss = F.mse_loss(rollout_data.returns, values_pred)

            # Entropy loss favor exploration
            if entropy is None:
                # Approximate entropy when no analytical form
                entropy_loss = -th.mean(-log_prob)
            else:
                entropy_loss = -th.mean(entropy)

            outer_loss = policy_loss + self.ent_coef * \
                entropy_loss + self.vf_coef * value_loss

            return outer_loss

    def train(self) -> None:
        """
        Update policy using the currently gathered
        rollout buffer.
        """
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(
                self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True

        # Meta opt
        meta_opt = th.optim.Adam(self.h_cell.parameters(), lr=1e-3)
        meta_opt.zero_grad()
        self.train_time += 1
        params_history = []
        # train for gradient_steps epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for _ in range(self.n_inner_loops):
                # Do a complete pass on the rollout buffer
                for rollout_data in self.rollout_buffer.get(self.batch_size):
                    actions = rollout_data.actions
                    if isinstance(self.action_space, spaces.Discrete):
                        # Convert discrete action from float to long
                        actions = rollout_data.actions.long().flatten()

                    # Re-sample the noise matrix because the log_std has changed
                    # if that line is commented (as in SAC)
                    if self.use_sde:
                        self.policy.reset_noise(self.batch_size)

                    if self.all_curricula:
                        values, log_prob, entropy = self.policy.evaluate_actions_all(rollout_data.observations,
                                                                                     rollout_data.memory,
                                                                                     rollout_data.subgoals,
                                                                                     rollout_data.shaping_rewards,
                                                                                     rollout_data.init_states,
                                                                                     actions)
                    elif self.goal_curriculum:
                        values, log_prob, entropy = self.policy.evaluate_actions_subgoal(rollout_data.observations,
                                                                                         rollout_data.subgoals,
                                                                                         actions)
                    elif self.memory_only:
                        values, log_prob, entropy = self.policy.evaluate_actions_memory(rollout_data.observations,
                                                                                        rollout_data.memory,
                                                                                        actions)
                    else:
                        values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations,
                                                                                 actions)
                    values = values.flatten()
                    # Normalize advantage
                    advantages = rollout_data.advantages
                    advantages = (advantages - advantages.mean()
                                  ) / (advantages.std() + 1e-8)

                    # ratio between old and new policy, should be one at the first iteration
                    ratio = th.exp(log_prob - rollout_data.old_log_prob)

                    # clipped surrogate loss
                    policy_loss_1 = advantages * ratio
                    policy_loss_2 = advantages * \
                        th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                    policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                    # Logging
                    pg_losses.append(policy_loss.item())
                    clip_fraction = th.mean(
                        (th.abs(ratio - 1) > clip_range).float()).item()
                    clip_fractions.append(clip_fraction)

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
                    value_loss = F.mse_loss(rollout_data.returns, values_pred)
                    value_losses.append(value_loss.item())

                    # Entropy loss favor exploration
                    if entropy is None:
                        # Approximate entropy when no analytical form
                        entropy_loss = -th.mean(-log_prob)
                    else:
                        entropy_loss = -th.mean(entropy)

                    entropy_losses.append(entropy_loss.item())

                    inner_loss = policy_loss + self.ent_coef * \
                        entropy_loss + self.vf_coef * value_loss
                    with th.no_grad():
                        log_ratio = log_prob - rollout_data.old_log_prob
                        approx_kl_div = th.mean(
                            (th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                        approx_kl_divs.append(approx_kl_div)

                    if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                        continue_training = False
                        if self.verbose >= 1:
                            print(
                                f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                        break
                    # Optimization step
                    self.policy.optimizer.zero_grad()
                    inner_loss.backward()
                    # Clip grad norm
                    th.nn.utils.clip_grad_norm_(
                        self.policy.parameters(), self.max_grad_norm)
                    self.policy.optimizer.step()
                    approx_kl_divs.append(
                        th.mean(rollout_data.old_log_prob - log_prob).detach().cpu().numpy())
                    params_history.append(self.policy.parameters())
                if not continue_training:
                    break

                # all_kl_divs.append(np.mean(approx_kl_divs))

                # if self.target_kl is not None and np.mean(approx_kl_divs) > 1.5 * self.target_kl:
                #     print(f"Early stopping at step {epoch} due to reaching max kl: {np.mean(approx_kl_divs):.2f}")
                #     break

                # update outer loss
                reverse_unroll(
                    params_history[-1], self.shared_hypernet.parameters(), self.outer_loss, set_grad=True)
                meta_opt.step()

        self._n_updates += self.n_epochs
        explained_var = explained_variance(
            self.rollout_buffer.returns.flatten(), self.rollout_buffer.values.flatten())

        logger.record("train/entropy_loss", np.mean(entropy_losses))
        logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        logger.record("train/value_loss", np.mean(value_losses))
        logger.record("train/approx_kl", np.mean(approx_kl_divs))
        logger.record("train/clip_fraction", np.mean(clip_fractions))
        logger.record("train/loss", inner_loss.item())
        logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            logger.record(
                "train/std", th.exp(self.policy.log_std).mean().item())

        logger.record("train/n_updates", self._n_updates,
                      exclude="tensorboard")
        logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            logger.record("train/clip_range_vf", clip_range_vf)

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
    ):

        return super(PPO, self).learn(
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

    # def save(
    #         self,
    #         path,
    #         exclude=None,
    #         include=None,
    # ) -> None:
    #     self.policy.save(path=path)

    def load(
            self,
            path,
            env: Optional[GymEnv] = None,
            device: Union[th.device, str] = "auto",
            **kwargs,
    ):
        self.policy.load(path=path)
