import random
import sys
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym
import numpy as np
import torch as th
from git import Object
from torch.nn import functional as F

from stable_baselines3.active_tamer.policies import ActiveSACHPolicy
from stable_baselines3.common.buffers import HumanReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.type_aliases import (
    GymEnv,
    MaybeCallback,
    RolloutReturn,
    Schedule,
    TrainFreq,
)
from stable_baselines3.common.utils import polyak_update, should_collect_more_steps
from stable_baselines3.common.vec_env import VecEnv
import time
from stable_baselines3.common.human_feedback import HumanFeedback
from stable_baselines3.common.online_learning_interface import FeedbackInterface

import os

class ActiveTamerRLSACRecord(OffPolicyAlgorithm):
    """
    TAMER + Soft Actor-Critic (SAC): Use trained SAC model to give feedback.
    Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor,
    This implementation borrows code from original implementation (https://github.com/haarnoja/sac)
    from OpenAI Spinning Up (https://github.com/openai/spinningup), from the softlearning repo
    (https://github.com/rail-berkeley/softlearning/)
    and from Stable Baselines (https://github.com/hill-a/stable-baselines)
    Paper: https://arxiv.org/abs/1801.01290
    Introduction to SAC: https://spinningup.openai.com/en/latest/algorithms/sac.html
    Note: we use double q target and not value target as discussed
    in https://github.com/hill-a/stable-baselines/issues/270
    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: learning rate for adam optimizer,
        the same learning rate will be used for all networks (Q-Values, Actor and Value function)
        it can be a function of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1)
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param action_noise: the action noise type (None by default), this can help
        for hard exploration problem. Cf common.noise for the different action noise type.
    :param replay_buffer_class: Replay buffer class to use (for instance ``HerReplayBuffer``).
        If ``None``, it will be automatically selected.
    :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param ent_coef: Entropy regularization coefficient. (Equivalent to
        inverse of reward scale in the original SAC paper.)  Controlling exploration/exploitation trade-off.
        Set it to 'auto' to learn it automatically (and 'auto_0.1' for using 0.1 as initial value)
    :param target_update_interval: update the target network every ``target_network_update_freq``
        gradient steps.
    :param target_entropy: target entropy when learning ``ent_coef`` (``ent_coef = 'auto'``)
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param use_sde_at_warmup: Whether to use gSDE instead of uniform sampling
        during the warm up phase (before learning starts)
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
        policy: Union[str, Type[ActiveSACHPolicy]],
        env: Union[GymEnv, str],
        trained_model = None,
        learning_rate: Union[float, Schedule] = 3e-4,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 5,
        batch_size: int = 2,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 1,
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[HumanReplayBuffer] = HumanReplayBuffer,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        ent_coef: Union[str, float] = "auto",
        target_update_interval: int = 1,
        target_entropy: Union[str, float] = "auto",
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        save_every: int = 2500,
        _init_setup_model: bool = True,
        model_name: str = "ActiveTamerRLSACRecord",
        render: bool = False,
        q_val_threshold: float = 0.999,
        rl_threshold: float = 0.1,
        prediction_threshold: float = 0.012,
        experiment_save_dir: str = "human_study/participant_default",
        scene_graph: Object = None,
        credit_assignment: int = 0,
    ):

        super(ActiveTamerRLSACRecord, self).__init__(
            policy,
            env,
            ActiveSACHPolicy,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            optimize_memory_usage=optimize_memory_usage,
            save_every=save_every,
            supported_action_spaces=(gym.spaces.Box),
            model_name=model_name,
            render=render,
        )

        self.target_entropy = target_entropy
        self.log_ent_coef = None  # type: Optional[th.Tensor]
        # Entropy coefficient / Entropy temperature
        # Inverse of the reward scale
        self.ent_coef = ent_coef
        self.target_update_interval = target_update_interval
        self.ent_coef_optimizer = None
        self.trained_model = trained_model
        self.curr_episode_timesteps = 0
        self.q_val_threshold = q_val_threshold
        self.rl_threshold = rl_threshold
        self.scene_graph = scene_graph
        self.prediction_threshold = prediction_threshold
        self.total_feedback = 0
        self.feedback_file = None
        if experiment_save_dir:
            os.makedirs(experiment_save_dir, exist_ok=True)
            self.feedback_file = open(os.path.join(experiment_save_dir, "feedback_file.txt"), "w")

        if _init_setup_model:
            self._setup_model()
        
        self.credit_assignment = credit_assignment

    def _setup_model(self) -> None:
        super(ActiveTamerRLSACRecord, self)._setup_model()
        self._create_aliases()
        # Target entropy is used when learning the entropy coefficient
        if self.target_entropy == "auto":
            # automatically set target entropy if needed
            self.target_entropy = -np.prod(self.env.action_space.shape).astype(
                np.float32
            )
        else:
            # Force conversion
            # this will also throw an error for unexpected string
            self.target_entropy = float(self.target_entropy)

        # The entropy coefficient or entropy can be learned automatically
        # see Automating Entropy Adjustment for Maximum Entropy RL section
        # of https://arxiv.org/abs/1812.05905
        if isinstance(self.ent_coef, str) and self.ent_coef.startswith("auto"):
            # Default initial value of ent_coef when learned
            init_value = 1.0
            if "_" in self.ent_coef:
                init_value = float(self.ent_coef.split("_")[1])
                assert (
                    init_value > 0.0
                ), "The initial value of ent_coef must be greater than 0"

            # Note: we optimize the log of the entropy coeff which is slightly different from the paper
            # as discussed in https://github.com/rail-berkeley/softlearning/issues/37
            self.log_ent_coef = th.log(
                th.ones(1, device=self.device) * init_value
            ).requires_grad_(True)
            self.ent_coef_optimizer = th.optim.Adam(
                [self.log_ent_coef], lr=self.lr_schedule(1)
            )
        else:
            # Force conversion to float
            # this will throw an error if a malformed string (different from 'auto')
            # is passed
            self.ent_coef_tensor = th.tensor(float(self.ent_coef)).to(self.device)

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target
        self.human_critic = self.policy.human_critic
        self.human_critic_target = self.policy.human_critic_target
        self.state_predictor = self.policy.state_predictor

    def train(
        self,
        gradient_steps: int,
        human_feedback_gui=None,
        batch_size: int = 64,
    ) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizers learning rate
        optimizers = [
            self.actor.optimizer,
            self.critic.optimizer,
            self.human_critic.optimizer,
        ]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses, human_critic_losses, state_prediction_losses = (
            [],
            [],
            [],
            [],
        )

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(
                batch_size, env=self._vec_normalize_env
            )

            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # Action by the current actor for the sampled state
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None:
                # Important: detach the variable from the graph
                # so we don't change it with other losses
                # see https://github.com/rail-berkeley/softlearning/issues/60
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(
                    self.log_ent_coef * (log_prob + self.target_entropy).detach()
                ).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            # Optimize entropy coefficient, also called
            # entropy temperature or alpha in the paper
            if ent_coef_loss is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                # Select action according to policy
                next_actions, next_log_prob = self.actor.action_log_prob(
                    replay_data.next_observations
                )
                # Compute the next Q values: min over all critics targets
                next_q_values = th.cat(
                    self.critic_target(replay_data.next_observations, next_actions),
                    dim=1,
                )
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                # add entropy term
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                # td error + entropy term
                target_q_values = (
                    replay_data.rewards
                    + (1 - replay_data.dones) * self.gamma * next_q_values
                )

                target_human_q_values = replay_data.humanRewards

            # Get current Q-values estimates for each critic network
            # using action from the replay buffer
            current_q_values = self.critic(
                replay_data.observations, replay_data.actions
            )

            # Compute critic loss
            critic_loss = 0.5 * sum(
                [
                    F.mse_loss(current_q, target_q_values)
                    for current_q in current_q_values
                ]
            )
            critic_losses.append(critic_loss.item())

            # Optimize the critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Get current Q-values estimates for human critic network
            # using action from the replay buffer
            current_human_q_values = self.human_critic(
                replay_data.observations, replay_data.actions
            )

            # Compute critic loss
            human_critic_loss = 0.5 * sum(
                [
                    F.mse_loss(current_q, target_human_q_values)
                    for current_q in current_human_q_values
                ]
            )
            human_critic_losses.append(human_critic_loss.item())

            # Optimize the critic
            self.human_critic.optimizer.zero_grad()
            human_critic_loss.backward()
            self.human_critic.optimizer.step()

            # Compute actor loss
            # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
            # Mean over all critic networks
            q_values_pi_critic = th.cat(
                self.critic.forward(replay_data.observations, actions_pi), dim=1
            )

            q_values_pi_human = th.cat(
                self.human_critic.forward(replay_data.observations, actions_pi), dim=1
            )

            q_values_pi = (
                self.rl_threshold * q_values_pi_critic
                + (1 - self.rl_threshold) * q_values_pi_human
            )

            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            actor_losses.append(actor_loss.item())

            if human_feedback_gui:
                human_feedback_gui.updateLoss(actor_loss.item())

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            predicted_states = self.state_predictor.forward(
                replay_data.observations, actions_pi.detach()
            )
            state_prediction_loss = F.mse_loss(
                predicted_states, replay_data.next_observations
            )
            self.state_predictor.optimizer.zero_grad()
            state_prediction_loss.backward()
            self.state_predictor.optimizer.step()
            state_prediction_losses.append(state_prediction_loss.item())

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(
                    self.critic.parameters(), self.critic_target.parameters(), self.tau
                )
                polyak_update(
                    self.human_critic.parameters(),
                    self.human_critic_target.parameters(),
                    self.tau,
                )

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record(
            "train/state_prediction_loss", np.mean(state_prediction_losses)
        )
        self.logger.record("train/human_critic_loss", np.mean(human_critic_losses))
        self.logger.record("train/rl_threshold", self.rl_threshold)
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

    def learn(
        self,
        total_timesteps: int,
        human_feedback_gui=None,
        human_feedback=None,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "ActiveTamerRLSACRecord",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> OffPolicyAlgorithm:

        return super(ActiveTamerRLSACRecord, self).learn(
            total_timesteps=total_timesteps,
            human_feedback_gui=human_feedback_gui,
            human_feedback=human_feedback,
            callback=callback,
            log_interval=log_interval,
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            tb_log_name=tb_log_name,
            eval_log_path=eval_log_path,
            reset_num_timesteps=reset_num_timesteps,
        )

    def _excluded_save_params(self) -> List[str]:
        return super(ActiveTamerRLSACRecord, self)._excluded_save_params() + [
            "actor",
            "critic",
            "critic_target",
            "human_critic",
            "human_critic_target",
        ]

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "actor.optimizer", "critic.optimizer"]
        if self.ent_coef_optimizer is not None:
            saved_pytorch_variables = ["log_ent_coef"]
            state_dicts.append("ent_coef_optimizer")
        else:
            saved_pytorch_variables = ["ent_coef_tensor"]
        return state_dicts, saved_pytorch_variables

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        train_freq: TrainFreq,
        replay_buffer: HumanReplayBuffer,
        action_noise: Optional[ActionNoise] = None,
        learning_starts: int = 0,
        log_interval: Optional[int] = None,
        human_feedback=None,
        human_feedback_gui=None,
    ) -> RolloutReturn:
        """
        Collect experiences and store them into a ``ReplayBuffer``.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param train_freq: How much experience to collect
            by doing rollouts of current policy.
            Either ``TrainFreq(<n>, TrainFrequencyUnit.STEP)``
            or ``TrainFreq(<n>, TrainFrequencyUnit.EPISODE)``
            with ``<n>`` being an integer greater than 0.
        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param replay_buffer:
        :param log_interval: Log data every ``log_interval`` episodes
        :return:
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        episode_rewards, total_timesteps = [], []
        num_collected_steps, num_collected_episodes = 0, 0
        curr_keyboard_feedback = None

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert env.num_envs == 1, "OffPolicyAlgorithm only support single environment"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."

        if self.use_sde:
            self.actor.reset_noise()

        callback.on_rollout_start()
        continue_training = True

        while should_collect_more_steps(
            train_freq, num_collected_steps, num_collected_episodes
        ):
            done = False
            episode_reward, episode_timesteps = 0.0, 0

            while not done:
                if (
                    self.use_sde
                    and self.sde_sample_freq > 0
                    and num_collected_steps % self.sde_sample_freq == 0
                ):
                    # Sample a new noise matrix
                    self.actor.reset_noise()

                # Select action randomly or according to policy
                action, buffer_action = self._sample_action(
                    learning_starts, action_noise
                )

                teacher_action, _ = self.trained_model.predict(self._last_obs)
                teacher_q_val = self.trained_model.critic.forward(
                    th.from_numpy(self._last_obs).to(self.device),
                    th.from_numpy(teacher_action).to(self.device),
                )
                teacher_q_val, _ = th.min(
                    th.cat(teacher_q_val, dim=1), dim=1, keepdim=True
                )
                teacher_q_val = teacher_q_val.cpu()[0][0]

                student_q_val = self.trained_model.critic.forward(
                    th.from_numpy(self._last_obs).to(self.device),
                    th.from_numpy(action).to(self.device),
                )
                student_q_val, _ = th.min(
                    th.cat(student_q_val, dim=1), dim=1, keepdim=True
                )
                student_q_val = student_q_val.cpu()[0][0]

                human_critic_qval_estimate = self.human_critic.forward(
                    th.from_numpy(self._last_obs).to(self.device),
                    th.from_numpy(action).to(self.device),
                )
                human_critic_qval_estimate, _ = th.min(
                    th.cat(human_critic_qval_estimate, dim=1), dim=1, keepdim=True
                )

                # Rescale and perform action
                if self.render:
                    env.render(mode="offscreen_robosuite")
                    # print("Current X and Y coord:" + str(self._last_obs.copy()))
                    # print("Next Action: " + str(action))

                self.logger.record("train/q_value_threshold", self.q_val_threshold)
                prev_obs = self._last_obs.copy()
                new_obs, reward, done, infos = env.step(action)
                self.feedback_file.write(
                    f"Current timestep = {str(self.num_timesteps)}. State = {str(new_obs)}. Action = {str(action)}. Reward = {str(reward)}\n"
                )
                self.feedback_file.write(
                    f"Curr episode timestep = {str(self.curr_episode_timesteps)}\n"
                )
                human_reward = 0
                state_prediction_err = F.mse_loss(
                    self.state_predictor(
                        th.from_numpy(prev_obs).to(self.device).reshape(1, -1),
                        th.from_numpy(action).to(self.device).reshape(1, -1),
                    ),
                    th.from_numpy(new_obs).to(self.device).reshape(1, -1),
                )
                scene_graph_updated, ucb_rank_high = self.scene_graph.updateGraph(new_obs, action)
                if (
                    # state_prediction_err > self.prediction_threshold
                    ucb_rank_high
                ):
                    self.feedback_file.write(
                        f"Scene graph at timestep {str(self.num_timesteps)} is {str(self.scene_graph.curr_graph)}\n"
                    )
                    self.feedback_file.write(
                        f"State prediction error at timestep {str(self.num_timesteps)} is {str(state_prediction_err)}\n"
                    )
                    if human_feedback:
                        _ = human_feedback.return_human_keyboard_feedback() # clear out buffer
                        curr_keyboard_feedback = (
                            human_feedback.return_human_keyboard_feedback()
                        )
                        simulated_human_reward = (
                            1
                            if self.q_val_threshold * teacher_q_val < student_q_val
                            else -1
                        )
                        print(f'Recommended Feedback at timestep {self.num_timesteps} is {str(simulated_human_reward)}')
                        while curr_keyboard_feedback is None or type(curr_keyboard_feedback) != int:
                            time.sleep(0.01)
                            curr_keyboard_feedback = (
                                human_feedback.return_human_keyboard_feedback()
                            ) # stall till you get human feedback
                            # print(f'{str(self.num_timesteps)}   {str(curr_keyboard_feedback)}')
                        human_reward = curr_keyboard_feedback
                        self.total_feedback += 1
                        self.scene_graph.updateRPE(human_reward, human_critic_qval_estimate)
                        self.feedback_file.write(
                            f"Human Feedback received at timestep {str(self.num_timesteps)} of {str(curr_keyboard_feedback)}\n"
                        )
                    
                    else:
                        raise "Must instantiate a human feedback object to collect human feedback."
                    
                self.q_val_threshold += 0.00000001
                self.num_timesteps += 1
                episode_timesteps += 1
                num_collected_steps += 1
                self.curr_episode_timesteps += 1

                self.logger.record(
                    "train/feedback_percentage",
                    self.total_feedback / self.num_timesteps,
                )
                # Give access to local variables
                callback.update_locals(locals())
                # Only stop training if return value is False, not when it is None.
                if callback.on_step() is False:
                    return RolloutReturn(
                        0.0,
                        num_collected_steps,
                        num_collected_episodes,
                        continue_training=False,
                    )

                # Retrieve reward and episode length if using Monitor wrapper
                self._update_info_buffer(infos, done)
                episode_reward += reward[0]

                # Store data in replay buffer (normalized action and unnormalized observation)
                self._store_transition(
                    replay_buffer,
                    buffer_action,
                    new_obs,
                    reward,
                    human_reward,
                    done,
                    infos,
                )

                # Can only do credit assignment for reward received from the environment
                if self.credit_assignment > 0:
                    self.apply_uniform_credit_assignment_human(
                        replay_buffer, float(human_reward), 0, min(self.credit_assignment, self.curr_episode_timesteps)
                    )

                if human_feedback_gui:
                    human_feedback_gui.updateReward(episode_reward)
                    human_feedback_gui.updateHumanReward(curr_keyboard_feedback)
                    human_feedback_gui.updateStateEstimation(state_prediction_err.item)

                self._update_current_progress_remaining(
                    self.num_timesteps, self._total_timesteps
                )

                # For DQN, check if the target network should be updated
                # and update the exploration schedule
                # For SAC/TD3, the update is done as the same time as the gradient update
                # see https://github.com/hill-a/stable-baselines/issues/900
                self._on_step()

                if not should_collect_more_steps(
                    train_freq, num_collected_steps, num_collected_episodes
                ):
                    break

            if done:
                self.curr_episode_timesteps = 0
                num_collected_episodes += 1
                self._episode_num += 1
                episode_rewards.append(episode_reward)
                total_timesteps.append(episode_timesteps)

                if action_noise is not None:
                    action_noise.reset()

                # Log training infos
                if log_interval is not None and self._episode_num % log_interval == 0:
                    self._dump_logs()

        mean_reward = np.mean(episode_rewards) if num_collected_episodes > 0 else 0.0

        callback.on_rollout_end()

        return RolloutReturn(
            mean_reward, num_collected_steps, num_collected_episodes, continue_training
        )

    def _store_transition(
        self,
        replay_buffer: HumanReplayBuffer,
        buffer_action: np.ndarray,
        new_obs: np.ndarray,
        reward: np.ndarray,
        human_reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        """
        Store transition in the replay buffer.
        We store the normalized action and the unnormalized observation.
        It also handles terminal observations (because VecEnv resets automatically).

        :param replay_buffer: Replay buffer object where to store the transition.
        :param buffer_action: normalized action
        :param new_obs: next observation in the current episode
            or first observation of the episode (when done is True)
        :param reward: reward for the current transition
        :param done: Termination signal
        :param infos: List of additional information about the transition.
            It may contain the terminal observations and information about timeout.
        """
        # Store only the unnormalized version
        if self._vec_normalize_env is not None:
            new_obs_ = self._vec_normalize_env.get_original_obs()
            reward_ = self._vec_normalize_env.get_original_reward()
        else:
            # Avoid changing the original ones
            self._last_original_obs, new_obs_, reward_ = self._last_obs, new_obs, reward

        # As the VecEnv resets automatically, new_obs is already the
        # first observation of the next episode
        if done and infos[0].get("terminal_observation") is not None:
            next_obs = infos[0]["terminal_observation"]
            # VecNormalize normalizes the terminal observation
            if self._vec_normalize_env is not None:
                next_obs = self._vec_normalize_env.unnormalize_obs(next_obs)
        else:
            next_obs = new_obs_

        replay_buffer.add(
            self._last_original_obs,
            next_obs,
            buffer_action,
            reward_,
            human_reward,
            done,
            infos,
        )

        self._last_obs = new_obs
        # Save the unnormalized observation
        if self._vec_normalize_env is not None:
            self._last_original_obs = new_obs_
