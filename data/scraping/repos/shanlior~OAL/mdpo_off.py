import time
import warnings

import gym

import numpy as np
import tensorflow as tf


from stable_baselines.common import dataset
from stable_baselines.common import tf_util, OffPolicyRLModel, SetVerbosity, TensorboardWriter
from stable_baselines.common.vec_env import VecEnv
from stable_baselines.common.math_util import safe_mean, unscale_action, scale_action
from stable_baselines.common.schedules import get_schedule_fn
from stable_baselines.common.buffers import ReplayBuffer
from stable_baselines.mdpo.policies import MDPOOFFPolicy
from stable_baselines import logger
from collections import deque



class MDPO_OFF(OffPolicyRLModel):
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
        Set it to 'auto' to learn it automatically (and 'auto_0.1' for using 0.1 as initial value),
        when using MDAL, this is not used
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
    :param seed: (int) Seed for the pseudo-random generators (python, numpy, tensorflow).
        If None (default), use random seed. Note that if you want completely deterministic
        results, you must set `n_cpu_tf_sess` to 1.
    :param n_cpu_tf_sess: (int) The number of threads for TensorFlow operations
        If None, the number of cpu of the current machine will be used.
    """

    def __init__(self, policy, env, gamma=0.99, learning_rate=3e-4, buffer_size=50000,
                 learning_starts=100, train_freq=1, update_reward_freq=2000, batch_size=256,
                 tau=0.005, ent_coef='auto', lam=0, target_update_interval=1,
                 gradient_steps=1, target_entropy='auto', action_noise=None,
                 random_exploration=0.0, verbose=0, tensorboard_log=None,
                 _init_setup_model=True, policy_kwargs=None, full_tensorboard_log=False,
                 seed=None, n_cpu_tf_sess=None, tsallis_q=1, reparameterize=True, t_pi=1.0, t_c=0.01,
                 timesteps_per_batch=1024, mdpo_update_steps=1):

        super(MDPO_OFF, self).__init__(policy=policy, env=env, replay_buffer=None, verbose=verbose,
                                  policy_base=MDPOOFFPolicy, requires_vec_env=False, policy_kwargs=policy_kwargs,
                                  seed=seed, n_cpu_tf_sess=n_cpu_tf_sess)

        self.using_mdal = False
        self.buffer_size = buffer_size
        self.learning_rate = learning_rate
        self.learning_starts = learning_starts
        self.train_freq = train_freq
        self.update_reward_freq = update_reward_freq
        self.update_reward_counter = 0
        self.batch_size = batch_size
        self.tau = tau
        self.timesteps_per_batch = timesteps_per_batch

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

        self.value_fn = None
        self.graph = None
        self.replay_buffer = None

        self.sess = None
        self.tensorboard_log = tensorboard_log
        self.verbose = verbose
        self.params = None
        self.summary = None
        self.policy_tf = None
        self.target_entropy = target_entropy
        self.full_tensorboard_log = full_tensorboard_log
        self.summary_cycle = 0
        self.summary_period = 10

        # GAIL Params
        self.hidden_size_adversary = 100
        self.adversary_entcoeff = 1e-3
        self.expert_dataset = None
        self.g_step = 1
        self.d_step = 1
        self.d_stepsize = 3e-4

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

        self.seed = seed
        self.lam = lam
        self.tsallis_q = tsallis_q
        self.reparameterize = reparameterize
        self.t_pi = t_pi
        self.t_c = t_c
        self.mdpo_update_steps = mdpo_update_steps


        if self.tsallis_q == 1:
            self.log_type = "log"
        else:
            self.log_type = "q-log"


        if _init_setup_model:
            self.setup_model()

    def _get_pretrain_placeholders(self):
        policy = self.policy_tf
        # Rescale
        deterministic_action = unscale_action(self.action_space, self.deterministic_action)
        return policy.obs_ph, self.actions_ph, deterministic_action

    def setup_model(self):

        # prevent import loops
        from stable_baselines.gail.adversary import TransitionClassifierMDPO
        from stable_baselines.mdal.adversary import TabularAdversaryTF, NeuralAdversary, NeuralAdversaryMDPO


        if isinstance(self.action_space, gym.spaces.Box):
            # Continuous action space
            self.discrete_actions = False
            self.n_actions = self.action_space.shape[0]
        elif isinstance(action_space, gym.spaces.Discrete):
            self.n_actions = self.action_space.n
            self.discrete_actions = True
        else:
            raise ValueError('Action space not supported: {}'.format(action_space))

        if self.is_action_features:
            self.n_features = self.n_actions + self.observation_space.shape[0]
        else:
            self.n_features = self.observation_space.shape[0]



        with SetVerbosity(self.verbose):
            self.graph = tf.Graph()
            with self.graph.as_default():
                self.set_random_seed(self.seed)
                # self.sess = tf_util.make_session(num_cpu=self.n_cpu_tf_sess, graph=self.graph)
                self.sess = tf_util.single_threaded_session(graph=self.graph)

                if self.using_gail:
                    self.reward_giver = TransitionClassifierMDPO(self.sess, self.observation_space, self.action_space,
                                                             self.hidden_size_adversary,
                                                             entcoeff=self.adversary_entcoeff,
                                                             lipschitz_reg_coef=self.lipschitz)
                elif self.using_mdal:
                    if self.neural:
                        self.reward_giver = NeuralAdversary(self.sess, self.observation_space, self.action_space,
                                                            self.hidden_size_adversary, normalize=True,
                                                            lipschitz_reg_coef=self.lipschitz)

                    else:
                        self.reward_giver = TabularAdversaryTF(self.sess, self.observation_space, self.action_space,
                                                                 self.hidden_size_adversary,
                                                                 entcoeff=self.adversary_entcoeff,
                                                                 expert_features=self.expert_dataset.successor_features,
                                                                 exploration_bonus=self.exploration_bonus,
                                                                 bonus_coef=self.bonus_coef,
                                                                 t_c=self.t_c,
                                                                 is_action_features=self.is_action_features)


                self.replay_buffer = ReplayBuffer(self.buffer_size, info=True)
                # if self.neural:
                    # self.traj_replay_buffer = ReplayBuffer(10, info=True)

                with tf.variable_scope("input", reuse=False):
                    # Create policy and target TF objects
                    self.policy_tf = self.policy(self.sess, self.observation_space, self.action_space,
                                                 **self.policy_kwargs)
                    self.target_policy = self.policy(self.sess, self.observation_space, self.action_space,
                                                     **self.policy_kwargs)
                    self.old_policy = self.policy(self.sess, self.observation_space, self.action_space,
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
                    self.ent_coef_ph = tf.placeholder(tf.float32, [], name="ent_coef_ph")


                with tf.variable_scope("model", reuse=False):
                    # Create the policy
                    # first return value corresponds to deterministic actions
                    # policy_out corresponds to stochastic actions, used for training
                    # logp_pi is the log probability of actions taken by the policy
                    # self.deterministic_action, policy_out, logp_pi = self.policy_tf.make_actor(self.processed_obs_ph)
                    self.deterministic_action, policy_out, logp_pi, policy_out_unsquashed = self.policy_tf.make_actor(self.processed_obs_ph, log_type=self.log_type, q_prime=self.tsallis_q, reparameterize=self.reparameterize)
                    self.policy_out = policy_out
                    self.policy_out_unsquashed = policy_out_unsquashed
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
                        self.target_entropy = -np.prod(self.action_space.shape).astype(np.float32)
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
                        # self.ent_coef = float(self.ent_coef)
                        self.ent_coef = self.ent_coef_ph

                with tf.variable_scope("target", reuse=False):
                    # Create the value network
                    _, _, value_target = self.target_policy.make_critics(self.processed_next_obs_ph,
                                                                         create_qf=False, create_vf=True)
                    self.value_target = value_target

                with tf.variable_scope("old", reuse=False):
                    _, self.policy_old_out, logp_pi_old, self.policy_old_out_unsquashed = self.old_policy.make_actor(self.processed_obs_ph, action=policy_out, log_type=self.log_type, q_prime=self.tsallis_q, reparameterize=self.reparameterize)
                    #_, _, logp_pi_buffer, _ = self.old_policy.make_actor(self.processed_obs_ph, action=self.actions_ph, log_type=self.log_type, q_prime=self.tsallis_q, reparameterize=self.reparameterize, reuse=True)

                    qf1_pi_old, _, _ = self.old_policy.make_critics(self.processed_obs_ph,
                                                                    policy_out, create_qf=True, create_vf=False)

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
                    qf1_loss = 0.5 * tf.reduce_mean((q_backup - qf1) ** 2)
                    qf2_loss = 0.5 * tf.reduce_mean((q_backup - qf2) ** 2)

                    # Compute the entropy temperature loss
                    # it is used when the entropy coefficient is learned
                    ent_coef_loss, entropy_optimizer = None, None
                    # if not isinstance(self.ent_coef, float):
                    #     ent_coef_loss = -tf.reduce_mean(
                    #         self.log_ent_coef * tf.stop_gradient(logp_pi + self.target_entropy))
                    #     entropy_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph)

                    # Compute the policy loss
                    # Alternative: policy_kl_loss = tf.reduce_mean(logp_pi - min_qf_pi)
                    if self.reparameterize:
                        policy_kl_loss = tf.reduce_mean(
                            logp_pi * self.ent_coef + (self.lam - self.ent_coef) * logp_pi_old - qf1_pi_old)
                        # policy_kl_loss = tf.reduce_mean(-logp_pi_old_action * tf.stop_gradient(tf.clip_by_value(tf.exp((logp_old * self.ent_coef + qf1_pi - value_fn) / (self.ent_coef + self.lam)), -10000, 10000)))
                    else:
                        # policy_kl_loss = tf.reduce_mean(-logp_pi_old_action * tf.stop_gradient(tf.exp((logp_old * self.ent_coef + qf1_pi - value_fn) / (self.ent_coef + self.lam))))
                        target_policy = tf.clip_by_value(
                            tf.exp((-logp_old * self.lam + qf1_pi - value_fn) / (self.ent_coef + self.lam)), 0, 0.9)
                        policy_kl_loss = tf.reduce_mean(target_policy * (target_policy - logp_pi_old_action))
                        # policy_kl_loss = tf.reduce_mean(-logp_pi * tf.stop_gradient(tf.exp((logp_pi_old * self.ent_coef + qf1_pi - value_fn) / self.ent_coef + self.lam)))

                    mean_reg_loss = 1e-3 * tf.reduce_mean(self.policy_tf.act_mu ** 2)
                    std_reg_loss = 1e-3 * tf.reduce_mean(self.policy_tf.logstd ** 2)
                    policy_reg_loss = mean_reg_loss + std_reg_loss #+ pre_activation_reg_loss

                    # policy_kl_loss = tf.reduce_mean(self.ent_coef * logp_pi - qf1_pi)

                    # NOTE: in the original implementation, they have an additional
                    # regularization loss for the Gaussian parameters
                    # this is not used for now
                    # policy_loss = (policy_kl_loss + policy_regularization_loss)
                    policy_loss = policy_kl_loss

                    # Target for value fn regression
                    # We update the vf towards the min of two Q-functions in order to
                    # reduce overestimation bias from function approximation error.
                    # v_backup = tf.stop_gradient(min_qf_pi - self.ent_coef * logp_pi)
                    v_backup = tf.stop_gradient(min_qf_pi - self.lam * logp_pi)

                    value_loss = 0.5 * tf.reduce_mean((value_fn - v_backup) ** 2)

                    values_losses = qf1_loss + qf2_loss + value_loss

                    # Policy train op
                    # (has to be separate from value train op, because min_qf_pi appears in policy_loss)
                    # policy_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph)
                    policy_optimizer = tf.train.AdamOptimizer(learning_rate=3e-4)
                    # grads, vars = zip(*policy_optimizer.compute_gradients(policy_loss, var_list=tf_util.get_trainable_vars('model/pi')))
                    # grads, norm = tf.clip_by_global_norm(grads, 100.0)
                    # policy_train_op = policy_optimizer.apply_gradients(zip(grads, vars))
                    policy_train_op = policy_optimizer.minimize(policy_loss, var_list=tf_util.get_trainable_vars('model/pi'))

                    # Value train op
                    # value_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph)
                    value_optimizer = tf.train.AdamOptimizer(learning_rate=3e-4)

                    # if self.using_mdal and self.neural:
                    #     self.reward_optimizer = tf.train.AdamOptimizer(learning_rate=3e-4)


                    values_params = tf_util.get_trainable_vars('model/values_fn')

                    source_params = tf_util.get_trainable_vars("model/values_fn")
                    target_params = tf_util.get_trainable_vars("target/values_fn")

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

                    curr_params = tf_util.get_trainable_vars("model/pi") + tf_util.get_trainable_vars("model/values_fn/qf1")
                    old_params = tf_util.get_trainable_vars("old/pi") + tf_util.get_trainable_vars("old/values_fn/qf1")
                    print("params", source_params, target_params, curr_params, old_params)

                    self.assign_policy_op = [
                        tf.assign(old, curr) #0.95 * old + (1 - 0.95) * curr)
                        for old, curr in zip(old_params, curr_params)
                    ]

                    # Control flow is used because sess.run otherwise evaluates in nondeterministic order
                    # and we first need to compute the policy action before computing q values losses
                    with tf.control_dependencies([policy_train_op]):
                        train_values_op = value_optimizer.minimize(values_losses, var_list=values_params)

                        self.infos_names = ['policy_loss', 'qf1_loss', 'qf2_loss', 'value_loss', 'entropy']
                        # All ops to call during one training step
                        # self.step_ops = [policy_loss, qf1_loss, qf2_loss,
                        #                  value_loss, qf1, qf2, value_fn, logp_pi,
                        #                  self.entropy, policy_train_op, train_values_op]
                        self.step_ops = [policy_loss, qf1_loss, qf2_loss,
                                         value_loss, qf1, qf2, value_fn, logp_pi,
                                         self.entropy, logp_pi_old, train_values_op, policy_train_op]

                        # Add entropy coefficient optimization operation if needed
                        if ent_coef_loss is not None:
                            with tf.control_dependencies([train_values_op]):
                                ent_coef_op = entropy_optimizer.minimize(ent_coef_loss, var_list=self.log_ent_coef)
                                self.infos_names += ['ent_coef_loss', 'ent_coef']
                                self.step_ops += [ent_coef_op, ent_coef_loss, self.ent_coef]

                    # if self.neural:
                    #     with tf.control_dependencies([reward_train_op]):
                    #         self.reward_train_op = [reward_train_op]


                    # Monitor losses and entropy in tensorboard
                    # tf.summary.scalar('policy_loss', policy_loss)
                    # tf.summary.scalar('qf1_loss', qf1_loss)
                    # tf.summary.scalar('qf2_loss', qf2_loss)
                    # tf.summary.scalar('value_loss', value_loss)
                    # tf.summary.scalar('entropy', self.entropy)
                    # if ent_coef_loss is not None:
                    #     tf.summary.scalar('ent_coef_loss', ent_coef_loss)
                    #     tf.summary.scalar('ent_coef', self.ent_coef)

                    # tf.summary.scalar('learning_rate', tf.reduce_mean(self.learning_rate_ph))



                # Retrieve parameters that must be saved
                self.params = tf_util.get_trainable_vars("model")
                if self.using_gail or (self.using_mdal and self.neural):
                    self.params.extend(self.reward_giver.get_trainable_variables())
                self.target_params = tf_util.get_trainable_vars("target/values_fn")

                # Initialize Variables and target network
                with self.sess.as_default():
                    self.sess.run(tf.global_variables_initializer())
                    self.sess.run(target_init_op)

                self.summary = tf.summary.merge_all()

    # def _train_step(self, step, writer, learning_rate, ent_coef):
    def _train_step(self, step, writer, learning_rate, kl_coeff):

        # Sample a batch from the replay buffer
        batch = self.replay_buffer.sample(self.batch_size, env=self._vec_normalize_env)
        batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones, batch_terminals = batch

        if self.using_mdal or self.using_gail:
            batch_rewards = self.reward_giver.get_reward(batch_obs, batch_actions)

        feed_dict = {
            self.observations_ph: batch_obs,
            self.actions_ph: batch_actions,
            self.next_observations_ph: batch_next_obs,
            self.rewards_ph: batch_rewards.reshape(self.batch_size, -1),
            self.terminals_ph: batch_terminals.reshape(self.batch_size, -1),
            self.learning_rate_ph: learning_rate,
            self.ent_coef_ph: kl_coeff

        }

        out = self.sess.run(self.step_ops, feed_dict)

        return

    def _initialize_dataloader(self):
        """Initialize dataloader."""
        batchsize = self.timesteps_per_batch // self.d_step
    #     batchsize = 1
    #     batchsize = self.batch_size
        self.expert_dataset.init_dataloader(batchsize)

    def learn(self, total_timesteps, callback=None,
              log_interval=2000, tb_log_name="MDPO_off", reset_num_timesteps=True, replay_wrapper=None):

        new_tb_log = self._init_num_timesteps(reset_num_timesteps)
        callback = self._init_callback(callback)

        if replay_wrapper is not None:
            self.replay_buffer = replay_wrapper(self.replay_buffer)
            # if self.neural:
            #     self.traj_replay_buffer = replay_wrapper(self.traj_replay_buffer)

        with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name, new_tb_log) \
                as writer:

            self._setup_learn()

            # Transform to callable if needed
            self.learning_rate = get_schedule_fn(self.learning_rate)
            # Initial learning rate
            current_lr = self.learning_rate(1)
            frac = 0
            t_pi = 0
            d_step = 0

            start_time = time.time()
            episode_rewards = [0.0]
            episode_true_rewards = [0.0]
            if self.using_mdal or self.using_gail:
                self._initialize_dataloader()
                true_reward_buffer = deque(maxlen=40)
                rewards_grad_norm_buffer = deque(maxlen=1)

            episode_successor_features = [np.zeros(self.n_features)]
            features_buffer = {}
            features_buffer['obs'], features_buffer['acs'], features_buffer['gammas'] = [[]], [[]], [[]]
            batch_buffer = {}
            batch_idx = 0
            batch_buffer['obs'], batch_buffer['acs'], batch_buffer['gammas'] = [], [], []

            episode_successes = []
            if self.action_noise is not None:
                self.action_noise.reset()
            obs = self.env.reset()
            h_step = 0
            # Retrieve unnormalized observation for saving into the buffer
            if self._vec_normalize_env is not None:
                obs_ = self._vec_normalize_env.get_original_obs().squeeze()
            else:
                obs_ = obs


            n_updates = 0
            infos_values = []

            callback.on_training_start(locals(), globals())
            callback.on_rollout_start()

            for step in range(total_timesteps):
                # Before training starts, randomly sample actions
                # from a uniform distribution for better exploration.
                # Afterwards, use the learned policy
                # if random_exploration is set to 0 (normal setting)
                if self.num_timesteps < self.learning_starts or np.random.rand() < self.random_exploration:
                    # actions sampled from action space are from range specific to the environment
                    # but algorithm operates on tanh-squashed actions therefore simple scaling is used
                    unscaled_action = self.env.action_space.sample()
                    action = scale_action(self.action_space, unscaled_action)
                else:
                    # if h_step < self.random_action_len:
                    #     unscaled_action = self.env.action_space.sample()
                    #     action = scale_action(self.action_space, unscaled_action)
                    # else:
                    action = self.policy_tf.step(obs[None], deterministic=False).flatten()
                    # Add noise to the action (improve exploration,
                    # not needed in general)
                    if self.action_noise is not None:
                        action = np.clip(action + self.action_noise(), -1, 1)
                    # inferred actions need to be transformed to environment action_space before stepping
                    unscaled_action = unscale_action(self.action_space, action)
                    assert action.shape == self.env.action_space.shape

                if self.using_mdal or self.using_gail:
                    # reward = reward_giver.get_reward(observation, (step+1) * covariance_lambda)
                    # covariance_lambda = (step+1) / (step + 2) * covariance_lambda \
                    #                     + np.matmul(np.expand_dims(observation, axis=1), np.expand_dims(observation, axis=0))\
                    #                     / (step + 2)
                    reward = self.reward_giver.get_reward(obs_, action)
                    new_obs, true_reward, done, info = self.env.step(unscaled_action)
                else:
                    new_obs, reward, done, info = self.env.step(unscaled_action)
                    true_reward = reward

                self.num_timesteps += 1

                # Only stop training if return value is False, not when it is None. This is for backwards
                # compatibility with callbacks that have no return statement.
                callback.update_locals(locals())
                if callback.on_step() is False:
                    break

                # Store only the unnormalized version
                if self._vec_normalize_env is not None:
                    new_obs_ = self._vec_normalize_env.get_original_obs().squeeze()
                    reward_ = self._vec_normalize_env.get_original_reward().squeeze()

                else:
                    # Avoid changing the original ones
                    obs_, new_obs_, reward_ = obs, new_obs, reward
                    true_reward_ = true_reward

                # Store transition in the replay buffer.
                self.replay_buffer_add(obs_, action, reward_, new_obs_, done, info)

                if len(batch_buffer['obs']) < self.update_reward_freq:
                    batch_buffer['obs'].append(obs_)
                    batch_buffer['acs'].append(action)
                    batch_buffer['gammas'].append(self.gamma ** h_step)

                else:
                    batch_buffer['obs'][batch_idx] = obs_
                    batch_buffer['acs'][batch_idx] = action
                    batch_buffer['gammas'][batch_idx] = self.gamma ** h_step

                    batch_idx = (batch_idx + 1) % self.update_reward_freq
                obs = new_obs
                # Save the unnormalized observation
                if self._vec_normalize_env is not None:
                    obs_ = new_obs_

                # Retrieve reward and episode length if using Monitor wrapper
                maybe_ep_info = info.get('episode')
                if maybe_ep_info is not None:
                    self.ep_info_buf.extend([maybe_ep_info])
                #
                if writer is not None:
                    # Write reward per episode to tensorboard
                    ep_reward = np.array([reward_]).reshape((1, -1))
                    ep_true_reward = np.array([true_reward_]).reshape((1, -1))
                    ep_done = np.array([done]).reshape((1, -1))
                    # tf_util.total_episode_reward_logger(self.episode_reward, ep_reward,
                    #                                     ep_done, writer, self.num_timesteps)
                    tf_util.total_episode_reward_logger(self.episode_reward, ep_true_reward,
                                                        ep_done, writer, self.num_timesteps)


                if self.num_timesteps % self.train_freq == 0:
                    callback.on_rollout_end()

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
                        t_pi = self.t_pi #* np.sqrt(step / total_timesteps)

                        # Update policy and critics (q functions)
                        # mb_infos_vals.append(self._train_step(step, writer, current_lr))
                        # mb_infos_vals.append(self._train_step(step, writer, current_lr, t_pi))
                        self._train_step(step, writer, current_lr, t_pi)
                        # Update target network
                        if (step + grad_step) % self.target_update_interval == 0:
                            # Update target network
                            self.sess.run(self.target_update_op)
                    # Log losses and entropy, useful for monitor training
                    # if len(mb_infos_vals) > 0:
                    #     infos_values = np.mean(mb_infos_vals, axis=0)

                if step % self.mdpo_update_steps == 0:
                    self.sess.run(self.assign_policy_op)

                # if step % (10 * self.update_reward_freq)  == 0:
                #     with self.sess.as_default():
                #         self.sess.run(self.reward_giver.update_old_rewards())


                # Update Rewards
                if (self.using_mdal or self.using_gail) and self.num_timesteps % self.update_reward_freq == 0\
                        and self.num_timesteps > 1 and self.num_timesteps > self.learning_starts:

                    if self.using_gail:
                        # ------------------ Update D ------------------
                        # logger.log("Optimizing Discriminator...")
                        # logger.log(fmt_row(13, self.reward_giver.loss_name))
                        # assert len(observation) == self.timesteps_per_batch
                        batch_size = self.timesteps_per_batch // self.d_step

                        # NOTE: uses only the last g step for observation
                        d_losses = []  # list of tuples, each of which gives the loss for a minibatch
                        # NOTE: for recurrent policies, use shuffle=False?

                        ob_batch, ac_batch = np.array(batch_buffer['obs']), np.array(batch_buffer['acs'])
                        ob_expert, ac_expert = self.expert_dataset.get_next_batch()
                        # update running mean/std for reward_giver

                        if self.reward_giver.normalize:
                            self.reward_giver.obs_rms.update(np.concatenate((ob_batch, ob_expert), 0))

                        # Reshape actions if needed when using discrete actions
                        if isinstance(self.action_space, gym.spaces.Discrete):
                            if len(ac_batch.shape) == 2:
                                ac_batch = ac_batch[:, 0]
                            if len(ac_expert.shape) == 2:
                                ac_expert = ac_expert[:, 0]

                        alpha = np.random.uniform(0.0, 1.0, size=(ob_batch.shape[0], 1))
                        ob_mix_batch = alpha * ob_batch + (1 - alpha) * ob_expert
                        ac_mix_batch = alpha * ac_batch + (1 - alpha) * ac_expert
                        with self.sess.as_default():
                            # self.reward_giver.train(ob_batch, ac_batch, np.expand_dims(gamma_batch, axis=1),
                            #                         ob_expert, ac_expert, np.expand_dims(gamma_expert, axis=1))
                            # if (step+1) % 10000 == 0:
                            #     self.sess.run(self.reward_giver.update_old_rewards())
                            # for _ in range(10):
                            self.reward_giver.train(ob_batch, ac_batch, ob_expert, ac_expert, ob_mix_batch, ac_mix_batch)
                        # with self.sess.as_default():
                        #     self.reward_giver.train(ob_batch, ac_batch, ob_expert, ac_expert)
                        # *newlosses, grad = self.reward_giver.lossandgrad(ob_batch, ac_batch, ob_expert, ac_expert)
                        # self.d_adam.update(self.allmean(grad), self.d_stepsize)
                        # d_losses.append(newlosses)
                        # logger.log(fmt_row(13, np.mean(d_losses, axis=0)))
                    elif self.using_mdal:
                        update_each_time = False
                        batch_sampling = True
                        if self.neural:
                            if batch_sampling:
                                if update_each_time:
                                    # frac = 1.0 - step / total_timesteps
                                    # current_lr = self.learning_rate(frac)
                                    # t_pi = self.t_pi  # * np.sqrt(step / total_timesteps)
                                    # batch = self.replay_buffer.sample(self.batch_size, env=self._vec_normalize_env)
                                    # ob_batch, ac_batch, _, _, _, _ = batch
                                    ob_batch, ac_batch = batch_buffer['obs'], batch_buffer['acs']
                                    ob_expert, ac_expert = self.expert_dataset.get_next_batch()
                                    if self.reward_giver.normalize:
                                        self.reward_giver.obs_rms.update(np.concatenate((ob_batch, ob_expert), 0))
                                    alpha = np.random.uniform(0.0, 1.0, size=(ob_batch.shape[0], 1))
                                    ob_mix_batch = alpha * ob_batch + (1 - alpha) * ob_expert
                                    ac_mix_batch = alpha * ac_batch + (1 - alpha) * ac_expert
                                    gamma_expert = gamma_batch = np.ones((ob_batch.shape[0]))

                                    with self.sess.as_default():
                                        self.reward_giver.train(ob_batch, ac_batch, np.expand_dims(gamma_batch, axis=1),
                                                                ob_expert, ac_expert, np.expand_dims(gamma_expert, axis=1),
                                                                ob_mix_batch, ac_mix_batch)
                                else:
                                    batch_size = self.timesteps_per_batch // self.d_step

                                    # NOTE: uses only the last g step for observation
                                    d_losses = []  # list of tuples, each of which gives the loss for a minibatch
                                    # NOTE: for recurrent policies, use shuffle=False?
                                    # with self.sess.as_default():
                                    #     self.sess.run(self.reward_giver.update_old_rewards())

                                    for ob_batch, ac_batch in dataset.iterbatches((batch_buffer['obs'], batch_buffer['acs']),
                                                                                  include_final_partial_batch=False,
                                                                                  batch_size=batch_size,
                                                                                  shuffle=True):
                                    # NOTE: uses only the last g step for observation
                                    # d_losses = []  # list of tuples, each of which gives the loss for a minibatch
                                    # NOTE: for recurrent policies, use shuffle=False?
                                    #
                                    # ob_batch, ac_batch, gamma_batch = np.array(batch_buffer['obs']), np.array(batch_buffer['acs']), np.array(batch_buffer['gammas'])
                                        ob_expert, ac_expert = self.expert_dataset.get_next_batch()
                                        gamma_batch = gamma_expert = np.ones((ob_expert.shape[0]))
                                        # ob_expert, ac_expert, gamma_expert = np.concatenate(self.expert_dataset.ep_obs),\
                                        #                                      np.concatenate(self.expert_dataset.ep_acs),\
                                        #                                      np.concatenate(self.expert_dataset.ep_gammas)
                                        # while True:
                                        #     ob_reg_expert, ac_reg_expert = self.expert_dataset.get_next_batch()
                                        #     ob_reg_expert, ac_reg_expert = np.array(ob_reg_expert), np.array(ac_reg_expert)
                                        #     # print("expert")
                                        #     # print(ob_reg_expert.shape)
                                        #     # print(ac_reg_expert.shape)
                                        #     if ob_reg_expert.shape[0] == ob_batch.shape[0] and ac_reg_expert.shape[0] == ac_batch.shape[0]:
                                        #         break
                                        # update running mean/std for reward_giver
                                        # if self.reward_giver.normalize:
                                        #     self.reward_giver.obs_rms.update(
                                        #         np.array(batch_successor_features)[:,:self.observation_space.shape[0]])
                                        if self.reward_giver.normalize:
                                            self.reward_giver.obs_rms.update(np.concatenate((ob_batch, ob_expert), 0))

                                        # Reshape actions if needed when using discrete actions
                                        if isinstance(self.action_space, gym.spaces.Discrete):
                                            if len(ac_batch.shape) == 2:
                                                ac_batch = ac_batch[:, 0]
                                            if len(ac_expert.shape) == 2:
                                                ac_expert = ac_expert[:, 0]
                                        alpha = np.random.uniform(0.0, 1.0, size=(ob_batch.shape[0], 1))
                                        ob_mix_batch = alpha * ob_batch + (1 - alpha) * ob_expert
                                        ac_mix_batch = alpha * ac_batch + (1 - alpha) * ac_expert
                                        with self.sess.as_default():
                                            # self.reward_giver.train(ob_batch, ac_batch, np.expand_dims(gamma_batch, axis=1),
                                            #                         ob_expert, ac_expert, np.expand_dims(gamma_expert, axis=1))
                                            # if (step+1) % 10000 == 0:
                                            #     self.sess.run(self.reward_giver.update_old_rewards())
                                            # for _ in range(10):
                                                self.reward_giver.train(ob_batch, ac_batch, np.expand_dims(gamma_batch, axis=1),
                                                                    ob_expert, ac_expert, np.expand_dims(gamma_expert, axis=1),
                                                                    ob_mix_batch, ac_mix_batch)
                                                # ob_expert, ac_expert = self.expert_dataset.get_next_batch()
                                                # alpha = np.random.uniform(0.0, 1.0, size=(ob_batch.shape[0], 1))
                                                # ob_mix_batch = alpha * ob_batch + (1 - alpha) * ob_reg_expert
                                                # ac_mix_batch = alpha * ac_batch + (1 - alpha) * ac_reg_expert
                                        # self.sess.run(self.reward_giver.update_old_rewards())
                                    # with self.sess.as_default():
                                    #     self.sess.run(self.reward_giver.update_old_rewards())

                                    # d_cycle_length = 10
                                    # d_step = (d_step + 1) % d_cycle_length
                                    #
                                    # if d_step == 0:
                                    # self.reward_giver.update_old_rewards()

                                # with self.sess.as_default():
                                #     self.reward_giver.train(ob_batch, ac_batch, np.ones((len(ob_batch), 1), dtype=np.float32),
                                #                             ob_expert, ac_expert, np.ones((len(ob_expert), 1), dtype=np.float32))
                            else:
                                # assert len(observation) == self.timesteps_per_batch
                                # Comment out if you want only the latest rewards:

                                ob_batch, ac_batch, gamma_batch = np.array(batch_buffer['obs']), np.array(
                                    batch_buffer['acs']), np.array(batch_buffer['gammas'])
                                ob_expert, ac_expert = self.expert_dataset.get_next_batch()
                                if self.reward_giver.normalize:
                                    self.reward_giver.obs_rms.update(np.concatenate((ob_batch, ob_expert), 0))

                                if self.update_reward_counter > -10:
                                    self.update_reward_counter = -10

                                batch_size = 100
                                num_updates = 1
                                for idx in range(num_updates):
                                    batch_steps = np.random.geometric(1 - self.gamma, batch_size)
                                    if h_step > 0:
                                        update_reward_counter_end = 0
                                    else:
                                        update_reward_counter_end = -1
                                    batch_episodes = np.random.randint(self.update_reward_counter,
                                                                       update_reward_counter_end, size=batch_size)

                                    ob_batch = np.zeros((batch_size, self.observation_space.shape[0]))
                                    ac_batch = np.zeros((batch_size, self.n_actions))

                                    expert_steps = np.random.geometric(1 - self.gamma, batch_size)
                                    expert_episodes = np.random.randint(self.expert_dataset.num_traj, size=batch_size)
                                    ob_expert = np.zeros((batch_size, self.observation_space.shape[0]))
                                    ac_expert = np.zeros((batch_size, self.n_actions))

                                    for i in range(batch_size):
                                        batch_episode = batch_episodes[i]
                                        ep_length = len(features_buffer['obs'][batch_episode])
                                        if ep_length > batch_steps[i]:
                                            batch_step = batch_steps[i]
                                        else:
                                            batch_step = ep_length - 1
                                        ob_batch[i] = np.array(features_buffer['obs'][batch_episode][batch_step])
                                        ac_batch[i] = np.array(features_buffer['acs'][batch_episode][batch_step])

                                        expert_episode = expert_episodes[i]
                                        expert_ep_length = len(self.expert_dataset.ep_obs[expert_episode])
                                        if expert_ep_length > expert_steps[i]:
                                            expert_step = expert_steps[i]
                                        else:
                                            expert_step = expert_ep_length - 1
                                        ob_expert[i] = np.array(self.expert_dataset.ep_obs[expert_episode][expert_step])
                                        ac_expert[i] = np.array(self.expert_dataset.ep_acs[expert_episode][expert_step])



                                    alpha = np.random.uniform(0.0, 1.0, size=(batch_size, 1))
                                    ob_mix_batch = alpha * ob_batch + (1 - alpha) * ob_expert
                                    ac_mix_batch = alpha * ac_batch + (1 - alpha) * ac_expert

                                    gamma_batch = np.ones((batch_size, 1))
                                    with self.sess.as_default():
                                        self.reward_giver.train(ob_batch, ac_batch, gamma_batch,
                                                                 ob_expert, ac_expert, gamma_batch,
                                                                ob_mix_batch, ac_mix_batch)
                            # else:
                            #     # assert len(observation) == self.timesteps_per_batch
                            #     # Comment out if you want only the latest rewards:
                            #     if self.update_reward_counter > -10:
                            #         self.update_reward_counter = -10
                            #         # np.random.geometric(self.gamma, 1000)
                            #
                            #     if done:
                            #         obs_batch, acs_batch, gammas_batch = \
                            #             features_buffer['obs'][self.update_reward_counter:],\
                            #             features_buffer['acs'][self.update_reward_counter:],\
                            #             features_buffer['gammas'][self.update_reward_counter:]
                            #         batch_successor_features = episode_successor_features[self.update_reward_counter:]
                            #
                            #     else:
                            #         obs_batch, acs_batch, gammas_batch = \
                            #             features_buffer['obs'][self.update_reward_counter:-1],\
                            #             features_buffer['acs'][self.update_reward_counter:-1],\
                            #             features_buffer['gammas'][self.update_reward_counter:-1]
                            #         batch_successor_features = episode_successor_features[self.update_reward_counter:-1]
                            #
                            #
                            #
                            #
                            #     if self.reward_giver.normalize:
                            #         ob_batch = batch_buffer['obs']
                            #         ob_expert, _ = self.expert_dataset.get_next_batch()
                            #         self.reward_giver.obs_rms.update(np.concatenate((ob_batch, ob_expert), 0))
                            #
                            #         # self.reward_giver.obs_rms.update(
                            #         #     np.array(batch_successor_features)[:, :self.observation_space.shape[0]])
                            #     # with self.sess.as_default():
                            #     #     self.reward_giver.zero_grad()
                            #     ob_reg_batch, ac_reg_batch = np.array(batch_buffer['obs']), np.array(batch_buffer['acs'])
                            #     while True:
                            #         ob_reg_expert, ac_reg_expert = self.expert_dataset.get_next_batch()
                            #         ob_reg_expert, ac_reg_expert = np.array(ob_reg_expert), np.array(ac_reg_expert)
                            #         # print("expert")
                            #         # print(ob_reg_expert.shape)
                            #         # print(ac_reg_expert.shape)
                            #         if ob_reg_expert.shape[0] == ob_reg_batch.shape[0] and ac_reg_expert.shape[0] == ac_reg_batch.shape[0]:
                            #             break
                            #
                            #     # print("batch")
                            #     # print(ob_reg_batch.shape)
                            #     # print(ac_reg_batch.shape)
                            #
                            #     for idx, (ob_batch, ac_batch, gamma_batch) in enumerate(zip(obs_batch, acs_batch, gammas_batch)):
                            #     # for idx in range(100):
                            #
                            #         #
                            #         # rand_traj = np.random.randint(self.expert_dataset.num_traj)
                            #         # ob_expert, ac_expert, gamma_expert = self.expert_dataset.ep_obs[rand_traj],\
                            #         #                                      self.expert_dataset.ep_acs[rand_traj],\
                            #         #                                      self.expert_dataset.ep_gammas[rand_traj]
                            #         #
                            #         #
                            #         # ob_batch, ac_batch, gamma_batch = np.array(ob_batch), np.array(ac_batch), np.array(gamma_batch)
                            #         # rand_traj_buffer = np.random.randint(len(obs_batch))
                            #         # ob_batch, ac_batch, gamma_batch = obs_batch[rand_traj_buffer], acs_batch[rand_traj_buffer], gammas_batch[rand_traj_buffer]
                            #         rand_traj_expert = np.random.randint(self.expert_dataset.num_traj)
                            #         # rand_start = np.random.randint(min(len(gamma_batch),len(self.expert_dataset.ep_obs[rand_traj_expert])))
                            #         rand_start = 0
                            #         ob_expert, ac_expert, gamma_expert = self.expert_dataset.ep_obs[rand_traj_expert][rand_start:],\
                            #                                              self.expert_dataset.ep_acs[rand_traj_expert][rand_start:],\
                            #                                              self.expert_dataset.ep_gammas[rand_traj_expert][rand_start:]
                            #         # ob_expert, ac_expert, gamma_expert = np.concatenate(self.expert_dataset.ep_obs),\
                            #         #                                      np.concatenate(self.expert_dataset.ep_acs), \
                            #         #                                      np.concatenate(self.expert_dataset.ep_gammas) / float(len(self.expert_dataset.ep_gammas))
                            #
                            #
                            #         ob_batch, ac_batch, gamma_batch = np.array(ob_batch[rand_start:]), np.array(ac_batch[rand_start:]), np.array(gamma_batch[rand_start:])
                            #
                            #         # with self.sess.as_default():
                            #         #     traj_len = 100
                            #         #     _, reward_grad_norm = self.reward_giver.train(ob_batch[:traj_len], ac_batch[:traj_len],
                            #         #                                                   np.expand_dims(gamma_batch, axis=1)[:traj_len],
                            #         #                                                   ob_expert[:traj_len], ac_expert[:traj_len],
                            #         #                                                   np.expand_dims(gamma_expert, axis=1)[:traj_len])
                            #         # with self.sess.as_default():
                            #         #     self.reward_giver.compute_grads(ob_batch, ac_batch, np.expand_dims(gamma_batch, axis=1) / (self.gamma ** rand_start),
                            #         #                              ob_expert, ac_expert, np.expand_dims(gamma_expert, axis=1) / (self.gamma ** rand_start))
                            #
                            #         alpha = np.random.uniform(0.0, 1.0, size=(ob_reg_batch.shape[0], 1))
                            #         ob_mix_batch = alpha * ob_reg_batch + (1 - alpha) * ob_reg_expert
                            #         ac_mix_batch = alpha * ac_reg_batch + (1 - alpha) * ac_reg_expert
                            #
                            #
                            #         with self.sess.as_default():
                            #             self.reward_giver.train(ob_batch, ac_batch, np.expand_dims(gamma_batch, axis=1),
                            #                                      ob_expert, ac_expert, np.expand_dims(gamma_expert, axis=1),
                            #                                     ob_mix_batch, ac_mix_batch)

                        else:
                            # policy_successor_features = np.mean(episode_successor_features[-21:-1], axis=0)
                            if done:
                                batch_successor_features = episode_successor_features[self.update_reward_counter:]
                            else:
                                batch_successor_features = episode_successor_features[self.update_reward_counter:-1]

                            policy_successor_features = np.mean(batch_successor_features, axis=0)
                            if self.reward_giver.normalize:
                                if self.reward_giver.is_action_features:
                                    self.reward_giver.obs_rms.update(np.array(batch_successor_features))
                                else:
                                    self.reward_giver.obs_rms.update(
                                        np.array(batch_successor_features)[:,:self.observation_space.shape[0]])

                            self.reward_giver.update_reward(policy_successor_features)


                        self.update_reward_counter = 0




                    callback.on_rollout_start()

                episode_rewards[-1] += reward_
                episode_true_rewards[-1] += true_reward_

                if done:

                    if self.action_noise is not None:
                        self.action_noise.reset()
                    if not isinstance(self.env, VecEnv):
                        obs = self.env.reset()
                    episode_rewards.append(0.0)
                    true_reward_buffer.extend([episode_true_rewards[-1]])
                    episode_true_rewards.append(0.0)

                    episode_successor_features.append(np.zeros(self.n_features))
                    features_buffer['obs'].append([])
                    features_buffer['acs'].append([])
                    features_buffer['gammas'].append([])

                    h_step = 0
                    self.update_reward_counter -= 1
                else:
                    if self.is_action_features:
                        concat_obs_ = np.concatenate((obs_, action), axis=0)
                    else:
                        concat_obs_ = obs_

                    episode_successor_features[-1] = np.add(episode_successor_features[-1],
                                                            (1 - self.gamma) * (self.gamma ** h_step) * concat_obs_)
                    features_buffer['obs'][-1].append(obs_)
                    features_buffer['acs'][-1].append(action)
                    features_buffer['gammas'][-1].append(self.gamma ** h_step)

                    h_step += 1

                    maybe_is_success = info.get('is_success')
                    if maybe_is_success is not None:
                        episode_successes.append(float(maybe_is_success))

                if len(episode_rewards[-101:-1]) == 0:
                    mean_reward = -np.inf
                    mean_true_reward = -np.inf
                else:
                    mean_reward = round(float(np.mean(episode_rewards[-101:-1])), 2)
                    mean_true_reward = round(float(np.mean(episode_true_rewards[-101:-1])), 1)


                # substract 1 as we appended a new term just now
                num_episodes = len(episode_rewards) - 1 
                # Display training infos
                # if self.verbose >= 1 and done and log_interval is not None and num_episodes % log_interval == 0:
                if self.verbose >= 1 and step % log_interval == 0: #done and log_interval is not None and len(episode_rewards) % log_interval == 0:
                    fps = int(step / (time.time() - start_time))
                    logger.logkv("steps", step)
                    logger.logkv("episodes", num_episodes)
                    logger.logkv("mean 100 episode reward", mean_reward)
                    logger.logkv("mean 100 episode true reward", mean_true_reward)
                    if len(self.ep_info_buf) > 0 and len(self.ep_info_buf[0]) > 0:
                        logger.logkv('EpTrueRewMean', safe_mean([ep_info['r'] for ep_info in self.ep_info_buf]))
                        # logger.logkv('EpTrueRewMean', safe_mean(true_reward_buffer))
                        logger.logkv('eplenmean', safe_mean([ep_info['l'] for ep_info in self.ep_info_buf]))
                    # logger.logkv('rewgradnorm', safe_mean(rewards_grad_norm_buffer))

                    logger.logkv("n_updates", n_updates)
                    logger.logkv("current_lr", current_lr)
                    # logger.logkv("ent_coef", 1-frac)
                    logger.logkv("fps", fps)
                    logger.logkv('time_elapsed', int(time.time() - start_time))
                    if len(episode_successes) > 0:
                        logger.logkv("success rate", np.mean(episode_successes[-100:]))
                    if len(infos_values) > 0:
                        for (name, val) in zip(self.infos_names, infos_values):
                            logger.logkv(name, val)
                    logger.logkv("total timesteps", self.num_timesteps)
                    logger.logkv("frac", frac)
                    logger.logkv("t_pi", t_pi)
                    logger.logkv("t_c", self.t_c)
                    logger.logkv("bonus_coef", self.bonus_coef)
                    logger.logkv("seed", self.seed)

                    logger.dumpkvs()
                    # Reset infos:
                    infos_values = []
            callback.on_training_end()
            return self

    def action_probability(self, observation, state=None, mask=None, actions=None, logp=False):
        if actions is not None:
            raise ValueError("Error: SAC does not have action probabilities.")

        warnings.warn("Even though SAC has a Gaussian policy, it cannot return a distribution as it "
                      "is squashed by a tanh before being scaled and outputed.")

        return None

    def predict(self, observation, state=None, mask=None, deterministic=True):
        observation = np.array(observation)
        vectorized_env = self._is_vectorized_observation(observation, self.observation_space)

        observation = observation.reshape((-1,) + self.observation_space.shape)
        actions = self.policy_tf.step(observation, deterministic=deterministic)
        actions = actions.reshape((-1,) + self.action_space.shape)  # reshape to the correct action shape
        actions = unscale_action(self.action_space, actions)  # scale the output for the prediction

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
            "update_reward_freq": self.update_reward_freq,
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
            "n_cpu_tf_sess": self.n_cpu_tf_sess,
            "seed": self.seed,
            "action_noise": self.action_noise,
            "random_exploration": self.random_exploration,
            "_vectorize_action": self._vectorize_action,
            "policy_kwargs": self.policy_kwargs
        }

        params_to_save = self.get_parameters()

        self._save_to_file(save_path, data=data, params=params_to_save, cloudpickle=cloudpickle)
