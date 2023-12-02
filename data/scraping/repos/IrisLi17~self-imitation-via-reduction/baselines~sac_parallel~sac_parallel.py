import sys
import time
import multiprocessing
from collections import deque
import warnings

import numpy as np
import tensorflow as tf

from stable_baselines.a2c.utils import total_episode_reward_logger
from stable_baselines.common import tf_util, OffPolicyRLModel, SetVerbosity, TensorboardWriter
from stable_baselines.common.vec_env import VecEnv
from utils.replay_buffer import MultiWorkerReplayBuffer, PrioritizedMultiWorkerReplayBuffer
from stable_baselines.ppo2.ppo2 import safe_mean, get_schedule_fn
from stable_baselines.sac.policies import SACPolicy
from stable_baselines import logger
from utils.eval_stack import eval_model


def get_vars(scope):
    """
    Alias for get_trainable_vars

    :param scope: (str)
    :return: [tf Variable]
    """
    return tf_util.get_trainable_vars(scope)


class SAC_parallel(OffPolicyRLModel):
    """
    Soft Actor-Critic (SAC)
    Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor,
    This implementation borrows code from original implementation (https://github.com/haarnoja/sac)
    from OpenAI Spinning Up (https://github.com/openai/spinningup) and from the Softlearning repo
    (https://github.com/rail-berkeley/softlearning/)
    Paper: https://arxiv.org/abs/1801.01290
    Introduction to SAC: https://spinningup.openai.com/en/latest/algorithms/sac.html

    :param policy: (SACPolicy or str) The policy model to use (MlpPolicy, CnnPolicy, LnMlpPolicy, ...)
    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param gamma: (float) the discount factor
    :param learning_rate: (float or callable) learning rate for adam optimizer,
        the same learning rate will be used for all networks (Q-Values, Actor and Value function)
        it can be a function of the current progress (from 1 to 0)
    :param buffer_size: (int) size of the replay buffer
    :param batch_size: (int) Minibatch size for each gradient update
    :param tau: (float) the soft update coefficient ("polyak update", between 0 and 1)
    :param ent_coef: (str or float) Entropy regularization coefficient. (Equivalent to
        inverse of reward scale in the original SAC paper.)  Controlling exploration/exploitation trade-off.
        Set it to 'auto' to learn it automatically (and 'auto_0.1' for using 0.1 as initial value)
    :param train_freq: (int) Update the model every `train_freq` steps.
    :param learning_starts: (int) how many steps of the model to collect transitions for before learning starts
    :param target_update_interval: (int) update the target network every `target_network_update_freq` steps.
    :param gradient_steps: (int) How many gradient update after each step
    :param target_entropy: (str or float) target entropy when learning ent_coef (ent_coef = 'auto')
    :param action_noise: (ActionNoise) the action noise type (None by default), this can help
        for hard exploration problem. Cf DDPG for the different action noise type.
    :param random_exploration: (float) Probability of taking a random action (as in an epsilon-greedy strategy)
        This is not needed for SAC normally but can help exploring when using HER + SAC.
        This hack was present in the original OpenAI Baselines repo (DDPG + HER)
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :param tensorboard_log: (str) the log location for tensorboard (if None, no logging)
    :param _init_setup_model: (bool) Whether or not to build the network at the creation of the instance
    :param policy_kwargs: (dict) additional arguments to be passed to the policy on creation
    :param full_tensorboard_log: (bool) enable additional logging when using tensorboard
        Note: this has no effect on SAC logging for now
    """

    def __init__(self, policy, env, gamma=0.99, learning_rate=3e-4, buffer_size=50000,
                 priority_buffer=False, alpha=0.6,
                 learning_starts=100, train_freq=1, batch_size=64,
                 tau=0.005, ent_coef='auto', target_update_interval=1,
                 gradient_steps=1, target_entropy='auto', action_noise=None,
                 eval_env=None, curriculum=False, sequential=False,
                 sil=False, sil_coef=1.0,
                 random_exploration=0.0, verbose=0, tensorboard_log=None,
                 _init_setup_model=True, policy_kwargs=None, full_tensorboard_log=False):

        super(SAC_parallel, self).__init__(policy=policy, env=env, replay_buffer=None, verbose=verbose,
                                           policy_base=SACPolicy, requires_vec_env=False, policy_kwargs=policy_kwargs)

        self.buffer_size = buffer_size
        self.priority_buffer = priority_buffer
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.learning_starts = learning_starts
        self.train_freq = train_freq
        self.batch_size = batch_size
        self.tau = tau
        # In the original paper, same learning rate is used for all networks
        # self.policy_lr = learning_rate
        # self.qf_lr = learning_rate
        # self.vf_lr = learning_rate
        # Entropy coefficient / Entropy temperature
        # Inverse of the reward scale
        self.ent_coef = ent_coef
        self.target_update_interval = target_update_interval
        self.gradient_steps = gradient_steps
        self.gamma = gamma
        self.action_noise = action_noise
        self.random_exploration = random_exploration
        self.eval_env = eval_env
        self.curriculum = curriculum
        self.sequential = sequential
        self.sil = sil
        self.sil_coef = sil_coef

        self.value_fn = None
        self.graph = None
        self.replay_buffer = None
        self.episode_reward = None
        self.sess = None
        self.tensorboard_log = tensorboard_log
        self.verbose = verbose
        self.params = None
        self.summary = None
        self.policy_tf = None
        self.target_entropy = target_entropy
        self.full_tensorboard_log = full_tensorboard_log

        self.obs_target = None
        self.target_policy = None
        self.actions_ph = None
        self.rewards_ph = None
        self.terminals_ph = None
        self.observations_ph = None
        self.action_target = None
        self.next_observations_ph = None
        self.value_target = None
        self.step_ops = None
        self.target_update_op = None
        self.infos_names = None
        self.entropy = None
        self.target_params = None
        self.learning_rate_ph = None
        self.processed_obs_ph = None
        self.processed_next_obs_ph = None
        self.log_ent_coef = None
        self.logpac_op = None

        if _init_setup_model:
            self.setup_model()

    def _get_pretrain_placeholders(self):
        policy = self.policy_tf
        # Rescale
        deterministic_action = self.deterministic_action * np.abs(self.action_space.low)
        return policy.obs_ph, self.actions_ph, deterministic_action

    def setup_model(self):
        with SetVerbosity(self.verbose):
            self.graph = tf.Graph()
            with self.graph.as_default():
                n_cpu = multiprocessing.cpu_count()
                if sys.platform == 'darwin':
                    n_cpu //= 2
                self.sess = tf_util.make_session(num_cpu=n_cpu, graph=self.graph)

                if hasattr(self.env, "env") and isinstance(self.env.env, VecEnv):
                    self.n_envs = self.env.env.num_envs
                else:
                    self.n_envs = 1

                if self.priority_buffer:
                    self.replay_buffer = PrioritizedMultiWorkerReplayBuffer(self.buffer_size, self.alpha,
                                                                            num_workers=self.env.env.num_envs,
                                                                            gamma=self.gamma)
                else:
                    self.replay_buffer = MultiWorkerReplayBuffer(self.buffer_size, num_workers=self.n_envs, gamma=self.gamma)

                with tf.variable_scope("input", reuse=False):
                    # Create policy and target TF objects
                    self.policy_tf = self.policy(self.sess, self.observation_space, self.action_space,
                                                 **self.policy_kwargs)
                    self.target_policy = self.policy(self.sess, self.observation_space, self.action_space,
                                                     **self.policy_kwargs)

                    # Initialize Placeholders
                    self.observations_ph = self.policy_tf.obs_ph
                    # Normalized observation for pixels
                    self.processed_obs_ph = self.policy_tf.processed_obs
                    self.next_observations_ph = self.target_policy.obs_ph
                    self.processed_next_obs_ph = self.target_policy.processed_obs
                    self.action_target = self.target_policy.action_ph
                    self.terminals_ph = tf.placeholder(tf.float32, shape=(None, 1), name='terminals')
                    self.rewards_ph = tf.placeholder(tf.float32, shape=(None, 1), name='rewards')
                    self.actions_ph = tf.placeholder(tf.float32, shape=(None,) + self.action_space.shape,
                                                     name='actions')
                    self.learning_rate_ph = tf.placeholder(tf.float32, [], name="learning_rate_ph")
                    self.importance_weight_ph = tf.placeholder(tf.float32, shape=(None,), name='weights')
                    # self.sum_rs_ph = tf.placeholder(tf.float32, shape=(None, 1), name="sum_rs")

                with tf.variable_scope("model", reuse=False):
                    # Create the policy
                    # first return value corresponds to deterministic actions
                    # policy_out corresponds to stochastic actions, used for training
                    # logp_pi is the log probabilty of actions taken by the policy
                    self.deterministic_action, policy_out, logp_pi = self.policy_tf.make_actor(self.processed_obs_ph)
                    # Monitor the entropy of the policy,
                    # this is not used for training
                    self.entropy = tf.reduce_mean(self.policy_tf.entropy)
                    #  Use two Q-functions to improve performance by reducing overestimation bias.
                    qf1, qf2, value_fn = self.policy_tf.make_critics(self.processed_obs_ph, self.actions_ph,
                                                                     create_qf=True, create_vf=True)
                    qf1_pi, qf2_pi, _ = self.policy_tf.make_critics(self.processed_obs_ph,
                                                                    policy_out, create_qf=True, create_vf=False,
                                                                    reuse=True)

                    # Target entropy is used when learning the entropy coefficient
                    if self.target_entropy == 'auto':
                        # automatically set target entropy if needed
                        self.target_entropy = -np.prod(self.env.action_space.shape).astype(np.float32)
                    else:
                        # Force conversion
                        # this will also throw an error for unexpected string
                        self.target_entropy = float(self.target_entropy)

                    # The entropy coefficient or entropy can be learned automatically
                    # see Automating Entropy Adjustment for Maximum Entropy RL section
                    # of https://arxiv.org/abs/1812.05905
                    if isinstance(self.ent_coef, str) and self.ent_coef.startswith('auto'):
                        # Default initial value of ent_coef when learned
                        init_value = 1.0
                        if '_' in self.ent_coef:
                            init_value = float(self.ent_coef.split('_')[1])
                            assert init_value > 0., "The initial value of ent_coef must be greater than 0"

                        self.log_ent_coef = tf.get_variable('log_ent_coef', dtype=tf.float32,
                                                            initializer=np.log(init_value).astype(np.float32))
                        self.ent_coef = tf.exp(self.log_ent_coef)
                    else:
                        # Force conversion to float
                        # this will throw an error if a malformed string (different from 'auto')
                        # is passed
                        self.ent_coef = float(self.ent_coef)

                with tf.variable_scope("target", reuse=False):
                    # Create the value network
                    _, _, value_target = self.target_policy.make_critics(self.processed_next_obs_ph,
                                                                         create_qf=False, create_vf=True)
                    self.value_target = value_target

                with tf.variable_scope("loss", reuse=False):
                    # Take the min of the two Q-Values (Double-Q Learning)
                    min_qf_pi = tf.minimum(qf1_pi, qf2_pi)

                    # Target for Q value regression
                    q_backup = tf.stop_gradient(
                        self.rewards_ph +
                        (1 - self.terminals_ph) * self.gamma * self.value_target
                    )

                    # Compute Q-Function loss
                    # TODO: test with huber loss (it would avoid too high values)
                    qf1_loss = 0.5 * tf.reduce_mean(self.importance_weight_ph * (q_backup - qf1) ** 2)
                    qf2_loss = 0.5 * tf.reduce_mean(self.importance_weight_ph * (q_backup - qf2) ** 2)

                    # Compute the entropy temperature loss
                    # it is used when the entropy coefficient is learned
                    ent_coef_loss, entropy_optimizer = None, None
                    if not isinstance(self.ent_coef, float):
                        ent_coef_loss = -tf.reduce_mean(
                            self.log_ent_coef * tf.stop_gradient(logp_pi + self.target_entropy))
                        entropy_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph)

                    # Compute the policy loss
                    # Alternative: policy_kl_loss = tf.reduce_mean(logp_pi - min_qf_pi)
                    policy_kl_loss = tf.reduce_mean(self.ent_coef * logp_pi - qf1_pi)
                    # Add optional SIL loss
                    if self.sil:
                        self.logpac_op = logp_ac = self.logpac(self.actions_ph)
                        policy_kl_loss += self.sil_coef * tf.reduce_mean(
                            -logp_ac * tf.stop_gradient(tf.nn.relu(qf1 - value_fn)))

                    # NOTE: in the original implementation, they have an additional
                    # regularization loss for the gaussian parameters
                    # this is not used for now
                    # policy_loss = (policy_kl_loss + policy_regularization_loss)
                    policy_loss = policy_kl_loss


                    # Target for value fn regression
                    # We update the vf towards the min of two Q-functions in order to
                    # reduce overestimation bias from function approximation error.
                    v_backup = tf.stop_gradient(min_qf_pi - self.ent_coef * logp_pi)
                    value_loss = 0.5 * tf.reduce_mean(self.importance_weight_ph * (value_fn - v_backup) ** 2)

                    values_losses = qf1_loss + qf2_loss + value_loss

                    # Policy train op
                    # (has to be separate from value train op, because min_qf_pi appears in policy_loss)
                    policy_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph)
                    policy_train_op = policy_optimizer.minimize(policy_loss, var_list=get_vars('model/pi'))

                    # Value train op
                    value_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph)
                    values_params = get_vars('model/values_fn')

                    source_params = get_vars("model/values_fn/vf")
                    target_params = get_vars("target/values_fn/vf")

                    # Polyak averaging for target variables
                    self.target_update_op = [
                        tf.assign(target, (1 - self.tau) * target + self.tau * source)
                        for target, source in zip(target_params, source_params)
                    ]
                    # Initializing target to match source variables
                    target_init_op = [
                        tf.assign(target, source)
                        for target, source in zip(target_params, source_params)
                    ]

                    # Control flow is used because sess.run otherwise evaluates in nondeterministic order
                    # and we first need to compute the policy action before computing q values losses
                    with tf.control_dependencies([policy_train_op]):
                        train_values_op = value_optimizer.minimize(values_losses, var_list=values_params)

                        self.infos_names = ['policy_loss', 'qf1_loss', 'qf2_loss', 'value_loss', 'entropy']
                        # All ops to call during one training step
                        self.step_ops = [policy_loss, qf1_loss, qf2_loss,
                                         value_loss, qf1, qf2, value_fn, logp_pi,
                                         self.entropy, policy_train_op, train_values_op]

                        # Add entropy coefficient optimization operation if needed
                        if ent_coef_loss is not None:
                            with tf.control_dependencies([train_values_op]):
                                ent_coef_op = entropy_optimizer.minimize(ent_coef_loss, var_list=self.log_ent_coef)
                                self.infos_names += ['ent_coef_loss', 'ent_coef']
                                self.step_ops += [ent_coef_op, ent_coef_loss, self.ent_coef]

                    # Monitor losses and entropy in tensorboard
                    tf.summary.scalar('policy_loss', policy_loss)
                    tf.summary.scalar('qf1_loss', qf1_loss)
                    tf.summary.scalar('qf2_loss', qf2_loss)
                    tf.summary.scalar('value_loss', value_loss)
                    tf.summary.scalar('entropy', self.entropy)
                    if ent_coef_loss is not None:
                        tf.summary.scalar('ent_coef_loss', ent_coef_loss)
                        tf.summary.scalar('ent_coef', self.ent_coef)

                    tf.summary.scalar('learning_rate', tf.reduce_mean(self.learning_rate_ph))

                # Retrieve parameters that must be saved
                self.params = get_vars("model")
                self.target_params = get_vars("target/values_fn/vf")

                # Initialize Variables and target network
                with self.sess.as_default():
                    self.sess.run(tf.global_variables_initializer())
                    self.sess.run(target_init_op)

                self.summary = tf.summary.merge_all()

    def _train_step(self, step, writer, learning_rate):
        # Sample a batch from the replay buffer
        if self.priority_buffer:
            batch = self.replay_buffer.sample(self.batch_size, beta=0.4)
            batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones, weights, idxes = batch
        else:
            batch = self.replay_buffer.sample(self.batch_size)
            batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones = batch
            weights = np.ones(batch_obs.shape[0])

        feed_dict = {
            self.observations_ph: batch_obs,
            self.actions_ph: batch_actions,
            self.next_observations_ph: batch_next_obs,
            self.rewards_ph: batch_rewards.reshape(self.batch_size, -1),
            self.terminals_ph: batch_dones.reshape(self.batch_size, -1),
            self.learning_rate_ph: learning_rate,
            self.importance_weight_ph: weights,
        }

        # Do one gradient step
        # and optionally compute log for tensorboard
        if writer is not None:
            out = self.sess.run([self.summary] + self.step_ops + [self.value_target], feed_dict)
            summary = out.pop(0)
            writer.add_summary(summary, step)
        else:
            out = self.sess.run(self.step_ops + [self.value_target], feed_dict)

        # Unpack to monitor losses and entropy
        policy_loss, qf1_loss, qf2_loss, value_loss, *values = out
        entropy = values[4]
        # Update priority
        if self.priority_buffer:
            qf1 = values[0]
            value_target = values[-1]
            batch_rewards = np.reshape(batch_rewards, (self.batch_size, -1))
            batch_dones = np.reshape(batch_dones, (self.batch_size, -1))
            priorities = batch_rewards + (1 - batch_dones) * self.gamma * value_target - qf1
            priorities = np.abs(priorities) + 1e-4
            priorities = np.squeeze(priorities, axis=-1).tolist()
            self.replay_buffer.update_priorities(idxes, priorities)

        if self.log_ent_coef is not None:
            ent_coef_loss, ent_coef = values[-3:-1]
            return policy_loss, qf1_loss, qf2_loss, value_loss, entropy, ent_coef_loss, ent_coef

        return policy_loss, qf1_loss, qf2_loss, value_loss, entropy

    def learn(self, total_timesteps, callback=None, seed=None,
              log_interval=4, tb_log_name="SAC", reset_num_timesteps=True, replay_wrapper=None):

        new_tb_log = self._init_num_timesteps(reset_num_timesteps)

        if replay_wrapper is not None:
            self.replay_buffer = replay_wrapper(self.replay_buffer)
            if self.priority_buffer:
                self.replay_buffer.set_model(self)

        with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name, new_tb_log) \
                as writer:

            self._setup_learn(seed)
            self.env_id = self.env.env.get_attr('spec')[0].id

            # Transform to callable if needed
            self.learning_rate = get_schedule_fn(self.learning_rate)
            # Initial learning rate
            current_lr = self.learning_rate(1)

            start_time = time.time()
            store_time = 0.0
            step_time = 0.0
            train_time = 0.0
            episode_rewards = [[0.0] for _ in range(self.env.env.num_envs)]
            episode_successes = [[] for _ in range(self.env.env.num_envs)]
            if self.action_noise is not None:
                self.action_noise.reset()
            assert isinstance(self.env.env, VecEnv)
            self.episode_reward = np.zeros((1,))
            ep_info_buf = deque(maxlen=100)
            n_updates = 0
            infos_values = []
            pp_sr_buf = deque(maxlen=5)
            stack_sr_buf = deque(maxlen=5)
            start_decay = total_timesteps
            if self.sequential and 'FetchStack' in self.env_id:
                current_max_nobject = 2
                self.env.env.env_method('set_task_array', [[(2, 0), (2, 1), (1, 0)]] * self.env.env.num_envs)
                print('Set task_array to ', self.env.env.get_attr('task_array')[0])
                self.env.env.env_method('set_random_ratio', [0.7] * self.env.env.num_envs)
            obs = self.env.reset()
            print(obs.shape)
            for step in range(total_timesteps):
                if callback is not None:
                    # Only stop training if return value is False, not when it is None. This is for backwards
                    # compatibility with callbacks that have no return statement.
                    if callback(locals(), globals()) is False:
                        break

                if self.curriculum and step % 3000 == 0:
                    if 'FetchStack' in self.env.env.get_attr('spec')[0].id:
                        # Stacking
                        pp_sr = eval_model(self.eval_env, self, current_max_nobject if self.sequential else self.env.env.get_attr('n_object')[0], 1.0,
                                           init_on_table=(self.env.env.get_attr('spec')[0].id == 'FetchStack-v2'))
                        pp_sr_buf.append(pp_sr)
                        stack_sr = eval_model(self.eval_env, self, current_max_nobject if self.sequential else self.env.env.get_attr('n_object')[0], 0.0,
                                              init_on_table=(self.env.env.get_attr('spec')[0].id == 'FetchStack-v2'))
                        stack_sr_buf.append(stack_sr)
                        print('Pick-and-place success rate', np.mean(pp_sr_buf))
                        if self.sequential:
                            if self.env.env.get_attr('random_ratio')[0] > 0.5 and np.mean(pp_sr_buf) > 0.8:
                                _ratio = 0.3
                            elif self.env.env.get_attr('random_ratio')[0] < 0.5 \
                                    and current_max_nobject < self.env.env.get_attr('n_object')[0] \
                                    and np.mean(stack_sr_buf) > 1 / current_max_nobject:
                                _ratio = 0.7
                                current_max_nobject += 1
                                previous_task_array = self.env.env.get_attr('task_array')[0]
                                self.env.env.env_method('set_task_array', [
                                    previous_task_array + [(current_max_nobject, j) for j in
                                                           range(current_max_nobject)]] * self.env.env.num_envs)

                                print('Set task_array to', self.env.env.get_attr('task_array')[0])
                            else:
                                _ratio = self.env.env.get_attr('random_ratio')[0]
                        else:
                            if start_decay == total_timesteps and np.mean(pp_sr_buf) > 0.8:
                                start_decay = step
                            _ratio = np.clip(0.7 - (step - start_decay) / 2e6, 0.3, 0.7)  # from 0.7 to 0.3
                    elif 'FetchPushWallObstacle' in self.env_id:
                        _ratio = max(1.0 - step / total_timesteps, 0.0)
                    else:
                        raise NotImplementedError
                    self.env.env.env_method('set_random_ratio', [_ratio] * self.env.env.num_envs)
                    print('Set random_ratio to', self.env.env.get_attr('random_ratio')[0])

                # Before training starts, randomly sample actions
                # from a uniform distribution for better exploration.
                # Afterwards, use the learned policy
                # if random_exploration is set to 0 (normal setting)
                if (self.num_timesteps < self.learning_starts
                    or np.random.rand() < self.random_exploration):
                    rescaled_action = np.stack([self.env.action_space.sample() for _ in range(self.env.env.num_envs)], axis=0)
                    action = rescaled_action
                else:
                    action = self.policy_tf.step(obs, deterministic=False)
                    # Add noise to the action (improve exploration,
                    # not needed in general)
                    if self.action_noise is not None:
                        action = np.clip(action + self.action_noise(), -1, 1)
                    # Rescale from [-1, 1] to the correct bounds
                    rescaled_action = action * np.abs(self.action_space.low)

                assert action.shape == (self.env.env.num_envs, ) + self.env.action_space.shape

                step_time0 = time.time()
                new_obs, reward, done, info = self.env.step(rescaled_action)
                step_time += time.time() - step_time0

                next_obs = new_obs.copy()
                for idx, _done in enumerate(done):
                    if _done:
                        next_obs[idx] = self.env.convert_dict_to_obs(info[idx]['terminal_observation'])

                # Store transition in the replay buffer.
                store_time0 = time.time()
                self.replay_buffer.add(obs, action, reward, next_obs, done)
                store_time += time.time() - store_time0
                obs = new_obs
                for idx, _done in enumerate(done):
                    episode_rewards[idx][-1] += reward[idx]
                    if _done:
                        episode_rewards[idx].append(0.0)
                        maybe_is_success = info[idx].get('is_success')
                        if maybe_is_success is not None:
                            episode_successes[idx].append(float(maybe_is_success))

                # Retrieve reward and episode length if using Monitor wrapper
                for _info in info:
                    maybe_ep_info = _info.get('episode')
                    if maybe_ep_info is not None:
                        ep_info_buf.extend([maybe_ep_info])

                if writer is not None:
                    # Write reward per episode to tensorboard
                    ep_reward = np.reshape(reward, (self.env.env.num_envs, -1))
                    ep_done = np.reshape(done, (self.env.env.num_envs, -1))
                    self.episode_reward = total_episode_reward_logger(self.episode_reward, ep_reward,
                                                                      ep_done, writer, self.num_timesteps)

                train_time0 = time.time()
                if step % self.train_freq == 0:
                    mb_infos_vals = []
                    # Update policy, critics and target networks
                    for grad_step in range(self.gradient_steps):
                        # Break if the warmup phase is not over
                        # or if there are not enough samples in the replay buffer
                        if not self.replay_buffer.can_sample(self.batch_size) \
                           or self.num_timesteps < self.learning_starts:
                            break
                        n_updates += 1
                        # Compute current learning_rate
                        frac = 1.0 - step / total_timesteps
                        current_lr = self.learning_rate(frac)
                        # Update policy and critics (q functions)
                        mb_infos_vals.append(self._train_step(step, writer, current_lr))
                        # Update target network
                        if (step + grad_step) % self.target_update_interval == 0:
                            # Update target network
                            self.sess.run(self.target_update_op)
                    # Log losses and entropy, useful for monitor training
                    if len(mb_infos_vals) > 0:
                        infos_values = np.mean(mb_infos_vals, axis=0)


                train_time += time.time() - train_time0
                if len(episode_rewards[0][-101:-1]) == 0:
                    mean_reward = -np.inf
                else:
                    mean_reward = round(float(np.mean(np.concatenate([episode_rewards[i][-101:-1]
                                                                      for i in range(self.env.env.num_envs)]))), 1)

                num_episodes = sum([len(episode_rewards[i]) for i in range(len(episode_rewards))])
                self.num_timesteps += self.env.env.num_envs
                # Display training infos
                if self.verbose >= 1 and done[0] and log_interval is not None and len(episode_rewards[0]) % (log_interval // self.env.env.num_envs) == 0:
                    fps = int(self.num_timesteps / (time.time() - start_time))
                    logger.logkv("episodes", num_episodes)
                    logger.logkv("mean 100 episode reward", mean_reward)
                    if len(ep_info_buf) > 0 and len(ep_info_buf[0]) > 0:
                        logger.logkv('ep_rewmean', safe_mean([ep_info['r'] for ep_info in ep_info_buf]))
                        logger.logkv('eplenmean', safe_mean([ep_info['l'] for ep_info in ep_info_buf]))
                    logger.logkv("n_updates", n_updates)
                    logger.logkv("current_lr", current_lr)
                    logger.logkv("fps", fps)
                    logger.logkv('time_elapsed', int(time.time() - start_time))
                    if len(episode_successes[0]) > 0:
                        logger.logkv("success rate", np.mean(np.concatenate([episode_successes[i][-100:] for i in range(self.env.env.num_envs)])))
                    if len(infos_values) > 0:
                        for (name, val) in zip(self.infos_names, infos_values):
                            logger.logkv(name, val)
                    logger.logkv("total timesteps", self.num_timesteps)
                    if hasattr(self.eval_env.unwrapped, 'random_ratio'):
                        logger.logkv("random_ratio", self.env.env.get_attr('random_ratio')[0])
                    logger.dumpkvs()
                    # Reset infos:
                    infos_values = []
            return self

    def logpac(self, action):
        from stable_baselines.sac.policies import gaussian_likelihood, EPS
        act_mu = self.policy_tf.act_mu
        log_std = tf.log(self.policy_tf.std)
        # Potentially we need to clip atanh and pass gradient
        log_u = gaussian_likelihood(tf.atanh(tf.clip_by_value(action, -0.99, 0.99)), act_mu, log_std)
        log_ac = log_u - tf.reduce_sum(tf.log(1 - action ** 2 + EPS), axis=1)
        return log_ac

    def action_probability(self, observation, state=None, mask=None, actions=None, logp=False):
        if actions is not None:
            raise ValueError("Error: SAC does not have action probabilities.")

        warnings.warn("Even though SAC has a Gaussian policy, it cannot return a distribution as it "
                      "is squashed by a tanh before being scaled and ouputed.")

        return None

    def predict(self, observation, state=None, mask=None, deterministic=True):
        observation = np.array(observation)
        vectorized_env = self._is_vectorized_observation(observation, self.observation_space)

        observation = observation.reshape((-1,) + self.observation_space.shape)
        actions = self.policy_tf.step(observation, deterministic=deterministic)
        actions = actions.reshape((-1,) + self.action_space.shape)  # reshape to the correct action shape
        actions = actions * np.abs(self.action_space.low)  # scale the output for the prediction

        if not vectorized_env:
            actions = actions[0]

        return actions, None

    def get_parameter_list(self):
        return (self.params +
                self.target_params)

    def save(self, save_path, cloudpickle=False):
        data = {
            "learning_rate": self.learning_rate,
            "buffer_size": self.buffer_size,
            "learning_starts": self.learning_starts,
            "train_freq": self.train_freq,
            "batch_size": self.batch_size,
            "tau": self.tau,
            "ent_coef": self.ent_coef if isinstance(self.ent_coef, float) else 'auto',
            "target_entropy": self.target_entropy,
            # Should we also store the replay buffer?
            # this may lead to high memory usage
            # with all transition inside
            # "replay_buffer": self.replay_buffer
            "gamma": self.gamma,
            "verbose": self.verbose,
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "policy": self.policy,
            "n_envs": self.n_envs,
            "action_noise": self.action_noise,
            "random_exploration": self.random_exploration,
            "_vectorize_action": self._vectorize_action,
            "policy_kwargs": self.policy_kwargs
        }

        params_to_save = self.get_parameters()

        self._save_to_file(save_path, data=data, params=params_to_save, cloudpickle=cloudpickle)
