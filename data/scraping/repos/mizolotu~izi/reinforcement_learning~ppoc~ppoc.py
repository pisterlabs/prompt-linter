import time

from reinforcement_learning import gym
import numpy as np
import tensorflow as tf

from threading import Thread
from collections import deque
from reinforcement_learning import logger
from reinforcement_learning.common import explained_variance, ActorCriticRLModel, tf_util, SetVerbosity, TensorboardWriter
from reinforcement_learning.common.runners import AbstractEnvRunner
from reinforcement_learning.common.policies import ActorCriticPolicy, RecurrentActorCriticPolicy, obs_autoencoder, inverse_model, forward_model
from reinforcement_learning.common.schedules import get_schedule_fn
from reinforcement_learning.common.tf_util import total_episode_reward_logger
from reinforcement_learning.common.math_util import safe_mean

#L2_WEIGHT = .1
L2_WEIGHT = 0.0

class PPOC(ActorCriticRLModel):
    """
    Proximal Policy Optimization algorithm (GPU version).
    Paper: https://arxiv.org/abs/1707.06347

    :param policy: (ActorCriticPolicy or str) The policy model to use (MlpPolicy, CnnPolicy, CnnLstmPolicy, ...)
    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param gamma: (float) Discount factor
    :param n_steps: (int) The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param ent_coef: (float) Entropy coefficient for the loss calculation
    :param learning_rate: (float or callable) The learning rate, it can be a function
    :param vf_coef: (float) Value function coefficient for the loss calculation
    :param max_grad_norm: (float) The maximum value for the gradient clipping
    :param lam: (float) Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param nminibatches: (int) Number of training minibatches per update. For recurrent policies,
        the number of environments run in parallel should be a multiple of nminibatches.
    :param noptepochs: (int) Number of epoch when optimizing the surrogate
    :param cliprange: (float or callable) Clipping parameter, it can be a function
    :param cliprange_vf: (float or callable) Clipping parameter for the value function, it can be a function.
        This is a parameter specific to the OpenAI implementation. If None is passed (default),
        then `cliprange` (that is used for the policy) will be used.
        IMPORTANT: this clipping depends on the reward scaling.
        To deactivate value function clipping (and recover the original PPO implementation),
        you have to pass a negative value (e.g. -1).
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :param tensorboard_log: (str) the log location for tensorboard (if None, no logging)
    :param _init_setup_model: (bool) Whether or not to build the network at the creation of the instance
    :param policy_kwargs: (dict) additional arguments to be passed to the policy on creation
    :param full_tensorboard_log: (bool) enable additional logging when using tensorboard
        WARNING: this logging can take a lot of space quickly
    :param seed: (int) Seed for the pseudo-random generators (python, numpy, tensorflow).
        If None (default), use random seed. Note that if you want completely deterministic
        results, you must set `n_cpu_tf_sess` to 1.
    :param n_cpu_tf_sess: (int) The number of threads for TensorFlow operations
        If None, the number of cpu of the current machine will be used.
    """
    def __init__(self, policy, env, gamma=0.99, n_steps=64, ent_coef=0.01, learning_rate=2.5e-4, vf_coef=0.5, n_runs=2,
                 max_grad_norm=0.5, lam=0.95, nminibatches=4, noptepochs=4, cliprange=0.2, cliprange_vf=None,
                 verbose=0, tensorboard_log='./tensorboard_log', _init_setup_model=True, policy_kwargs=None,
                 full_tensorboard_log=False, seed=None, n_cpu_tf_sess=None, beta=0.2, lmd=0.1, eta=0.01):

        #tensorboard_log = None

        self.learning_rate = learning_rate
        self.cliprange = cliprange
        self.cliprange_vf = cliprange_vf
        self.n_steps = n_steps
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.gamma = gamma
        self.lam = lam
        self.nminibatches = nminibatches
        self.noptepochs = noptepochs
        self.tensorboard_log = tensorboard_log
        self.full_tensorboard_log = full_tensorboard_log

        self.n_runs = n_runs

        self.action_ph = None
        self.advs_ph = None
        self.rewards_ph = None
        self.old_neglog_pac_ph = None
        self.old_vpred_ph = None
        self.learning_rate_ph = None
        self.clip_range_ph = None
        self.entropy = None
        self.vf_loss = None
        self.pg_loss = None
        self.approxkl = None
        self.clipfrac = None
        self._train = None
        self.loss_names = None
        self.train_model = None
        self.act_model = None
        self.value = None
        self.n_batch = None
        self.summary = None

        self.beta = beta
        self.lmd = lmd
        self.eta = eta

        super().__init__(policy=policy, env=env, verbose=verbose, requires_vec_env=True,
                         _init_setup_model=_init_setup_model, policy_kwargs=policy_kwargs,
                         seed=seed, n_cpu_tf_sess=n_cpu_tf_sess)

        if _init_setup_model:
            self.setup_model()

    def _make_runner(self):
        return Runner(env=self.env, model=self, n_runs=self.n_runs, n_steps=self.n_steps, gamma=self.gamma, lam=self.lam)

    def _get_pretrain_placeholders(self):
        policy = self.act_model
        if isinstance(self.action_space, gym.spaces.Discrete):
            return policy.obs_ph, self.action_ph, policy.policy
        return policy.obs_ph, self.action_ph, policy.deterministic_action

    def setup_model(self):
        with SetVerbosity(self.verbose):

            assert issubclass(self.policy, ActorCriticPolicy), "Error: the input policy for the PPO2 model must be " \
                                                               "an instance of common.policies.ActorCriticPolicy."

            self.n_batch = self.n_envs * self.n_runs * self.n_steps

            self.graph = tf.Graph()
            with self.graph.as_default():
                self.set_random_seed(self.seed)
                self.sess = tf_util.make_session(num_cpu=self.n_cpu_tf_sess, graph=self.graph)

                n_batch_step = None
                n_batch_train = None
                if issubclass(self.policy, RecurrentActorCriticPolicy):
                    assert self.n_envs % self.nminibatches == 0, "For recurrent policies, "\
                        "the number of environments run in parallel should be a multiple of nminibatches."
                    n_batch_step = self.n_envs
                    n_batch_train = self.n_batch // self.nminibatches

                act_model = self.policy(self.sess, self.observation_space, self.action_space, self.n_envs, 1, n_batch_step, reuse=False, **self.policy_kwargs)
                with tf.compat.v1.variable_scope("train_model", reuse=True, custom_getter=tf_util.outer_scope_getter("train_model")):
                    train_model = self.policy(self.sess, self.observation_space, self.action_space, self.n_envs * self.n_runs // self.nminibatches, self.n_steps, n_batch_train, reuse=True, **self.policy_kwargs)

                self.observation_ph = tf.compat.v1.placeholder(shape=(None,) + self.observation_space.shape, dtype=self.observation_space.dtype, name='obs')
                self.processed_obs = tf.cast(self.observation_ph, tf.float32)

                self.observation_next_ph = tf.compat.v1.placeholder(shape=(None,) + self.observation_space.shape, dtype=self.observation_space.dtype, name='obs_next')
                self.processed_obs_next = tf.cast(self.observation_next_ph, tf.float32)

                with tf.compat.v1.variable_scope("obs_encoded", reuse=tf.compat.v1.AUTO_REUSE):
                    self.obs_encoded = obs_autoencoder(self.processed_obs, self.observation_space)
                    self.obs_next_encoded = obs_autoencoder(self.processed_obs_next, self.observation_space)
                #self.obs_encoded = self.processed_obs
                #self.obs_next_encoded = self.processed_obs_next

                self.act_hat = inverse_model(self.obs_encoded, self.obs_next_encoded, self.action_space)

                with tf.compat.v1.variable_scope("loss", reuse=False):
                    self.action_ph = train_model.pdtype.sample_placeholder([None], name="action_ph")

                    self.processed_act = tf.cast(tf.one_hot(self.action_ph, self.action_space.n), tf.float32)
                    self.obs_next_hat = forward_model(self.obs_encoded, self.processed_act, self.observation_space)

                    self.advs_ph = tf.compat.v1.placeholder(tf.float32, [None], name="advs_ph")
                    self.rewards_ph = tf.compat.v1.placeholder(tf.float32, [None], name="rewards_ph")

                    self.true_rewards_ph = tf.compat.v1.placeholder(tf.float32, [None], name="true_rewards_ph")

                    self.old_neglog_pac_ph = tf.compat.v1.placeholder(tf.float32, [None], name="old_neglog_pac_ph")
                    self.old_vpred_ph = tf.compat.v1.placeholder(tf.float32, [None], name="old_vpred_ph")
                    self.learning_rate_ph = tf.compat.v1.placeholder(tf.float32, [], name="learning_rate_ph")
                    self.clip_range_ph = tf.compat.v1.placeholder(tf.float32, [], name="clip_range_ph")
                    neglogpac = train_model.proba_distribution.neglogp(self.action_ph)
                    self.entropy = tf.reduce_mean(input_tensor=train_model.proba_distribution.entropy())
                    vpred = train_model.value_flat

                    # Value function clipping: not present in the original PPO
                    if self.cliprange_vf is None:
                        # Default behavior (legacy from OpenAI baselines):
                        # use the same clipping as for the policy
                        self.clip_range_vf_ph = self.clip_range_ph
                        self.cliprange_vf = self.cliprange
                    elif isinstance(self.cliprange_vf, (float, int)) and self.cliprange_vf < 0:
                        # Original PPO implementation: no value function clipping
                        self.clip_range_vf_ph = None
                    else:
                        # Last possible behavior: clipping range
                        # specific to the value function
                        self.clip_range_vf_ph = tf.compat.v1.placeholder(tf.float32, [], name="clip_range_vf_ph")

                    if self.clip_range_vf_ph is None:
                        # No clipping
                        vpred_clipped = train_model.value_flat
                    else:
                        # Clip the different between old and new value
                        # NOTE: this depends on the reward scaling
                        vpred_clipped = self.old_vpred_ph + tf.clip_by_value(train_model.value_flat - self.old_vpred_ph, - self.clip_range_vf_ph, self.clip_range_vf_ph)

                    vf_losses1 = tf.square(vpred - self.rewards_ph)
                    vf_losses2 = tf.square(vpred_clipped - self.rewards_ph)
                    self.vf_loss = .5 * tf.reduce_mean(input_tensor=tf.maximum(vf_losses1, vf_losses2))

                    ratio = tf.exp(self.old_neglog_pac_ph - neglogpac)
                    pg_losses = -self.advs_ph * ratio
                    pg_losses2 = -self.advs_ph * tf.clip_by_value(ratio, 1.0 - self.clip_range_ph, 1.0 + self.clip_range_ph)
                    self.pg_loss = tf.reduce_mean(input_tensor=tf.maximum(pg_losses, pg_losses2))
                    self.approxkl = .5 * tf.reduce_mean(input_tensor=tf.square(neglogpac - self.old_neglog_pac_ph))
                    self.clipfrac = tf.reduce_mean(input_tensor=tf.cast(tf.greater(tf.abs(ratio - 1.0), self.clip_range_ph), tf.float32))

                    self.params = tf.compat.v1.trainable_variables()
                    weight_params = [v for v in self.params if '/b' not in v.name]
                    l2_loss = tf.reduce_sum([tf.nn.l2_loss(v) for v in weight_params])

                    self.frw_loss = 0.5 * tf.reduce_sum(tf.math.square(self.obs_next_encoded - self.obs_next_hat))

                    #self.inv_loss = - tf.reduce_sum(self.processed_act * tf.math.log(self.act_hat + tf.keras.backend.epsilon()))

                    self.inv_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.act_hat, labels=tf.cast(self.action_ph, tf.int64)))

                    self.int_loss = self.beta * self.frw_loss + (1.0 - self.beta) * self.inv_loss
                    loss = self.lmd * (self.pg_loss - self.entropy * self.ent_coef + self.vf_loss * self.vf_coef) + self.int_loss

                    self.int_reward = self.eta * self.frw_loss

                    tf.compat.v1.summary.scalar('entropy_loss', self.entropy)
                    tf.compat.v1.summary.scalar('policy_gradient_loss', self.pg_loss)
                    tf.compat.v1.summary.scalar('value_function_loss', self.vf_loss)
                    tf.compat.v1.summary.scalar('intrinsic_loss', self.int_loss)
                    tf.compat.v1.summary.scalar('approximate_kullback-leibler', self.approxkl)
                    tf.compat.v1.summary.scalar('clip_factor', self.clipfrac)
                    tf.compat.v1.summary.scalar('loss', loss)

                    for var in self.params:
                        print(var.name, var)

                    with tf.compat.v1.variable_scope('model'):
                        if self.full_tensorboard_log:
                            for var in self.params:
                                tf.compat.v1.summary.histogram(var.name, var)
                    grads = tf.gradients(ys=loss, xs=self.params)
                    if self.max_grad_norm is not None:
                        grads, _grad_norm = tf.clip_by_global_norm(grads, self.max_grad_norm)
                    grads = list(zip(grads, self.params))
                    for gr in grads:
                        print(gr)
                trainer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate_ph, epsilon=1e-5)
                self._train = trainer.apply_gradients(grads)

                self.loss_names = ['policy_loss', 'value_loss', 'int_loss', 'policy_entropy', 'approxkl', 'clipfrac']

                with tf.compat.v1.variable_scope("input_info", reuse=False):
                    tf.compat.v1.summary.scalar('true_rewards', tf.reduce_mean(input_tensor=self.true_rewards_ph))
                    tf.compat.v1.summary.scalar('discounted_rewards', tf.reduce_mean(input_tensor=self.rewards_ph))
                    tf.compat.v1.summary.scalar('learning_rate', tf.reduce_mean(input_tensor=self.learning_rate_ph))
                    tf.compat.v1.summary.scalar('advantage', tf.reduce_mean(input_tensor=self.advs_ph))
                    tf.compat.v1.summary.scalar('clip_range', tf.reduce_mean(input_tensor=self.clip_range_ph))
                    if self.clip_range_vf_ph is not None:
                        tf.compat.v1.summary.scalar('clip_range_vf', tf.reduce_mean(input_tensor=self.clip_range_vf_ph))

                    tf.compat.v1.summary.scalar('old_neglog_action_probability', tf.reduce_mean(input_tensor=self.old_neglog_pac_ph))
                    tf.compat.v1.summary.scalar('old_value_pred', tf.reduce_mean(input_tensor=self.old_vpred_ph))

                    if self.full_tensorboard_log:
                        tf.compat.v1.summary.histogram('discounted_rewards', self.rewards_ph)
                        tf.compat.v1.summary.histogram('learning_rate', self.learning_rate_ph)
                        tf.compat.v1.summary.histogram('advantage', self.advs_ph)
                        tf.compat.v1.summary.histogram('clip_range', self.clip_range_ph)
                        tf.compat.v1.summary.histogram('old_neglog_action_probability', self.old_neglog_pac_ph)
                        tf.compat.v1.summary.histogram('old_value_pred', self.old_vpred_ph)
                        if tf_util.is_image(self.observation_space):
                            tf.compat.v1.summary.image('observation', train_model.obs_ph)
                        else:
                            tf.compat.v1.summary.histogram('observation', train_model.obs_ph)

                self.train_model = train_model
                self.act_model = act_model
                self.step = act_model.step
                self.proba_step = act_model.proba_step
                self.value = act_model.value
                self.initial_state = act_model.initial_state
                tf.compat.v1.global_variables_initializer().run(session=self.sess)  # pylint: disable=E1101

                self.summary = tf.compat.v1.summary.merge_all()

    def intrinsic_reward(self, obs, obs_next, actions):
        r = self.sess.run([self.int_reward], {
            self.observation_ph: obs, self.observation_next_ph: obs_next, self.action_ph: actions,
        })[0]
        return r

    def _train_step(self, learning_rate, cliprange, obs, obs_next, returns, true_rewards, masks, actions, values, neglogpacs, update, writer, states=None, cliprange_vf=None):
        """
        Training of PPO2 Algorithm

        :param learning_rate: (float) learning rate
        :param cliprange: (float) Clipping factor
        :param obs: (np.ndarray) The current observation of the environment
        :param returns: (np.ndarray) the rewards
        :param masks: (np.ndarray) The last masks for done episodes (used in recurent policies)
        :param actions: (np.ndarray) the actions
        :param values: (np.ndarray) the values
        :param neglogpacs: (np.ndarray) Negative Log-likelihood probability of Actions
        :param update: (int) the current step iteration
        :param writer: (TensorFlow Summary.writer) the writer for tensorboard
        :param states: (np.ndarray) For recurrent policies, the internal state of the recurrent model
        :return: policy gradient loss, value function loss, policy entropy,
                approximation of kl divergence, updated clipping range, training update operation
        :param cliprange_vf: (float) Clipping factor for the value function
        """
        advs = returns - values
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        td_map = {
            self.train_model.obs_ph: obs, self.action_ph: actions,
            self.advs_ph: advs, self.rewards_ph: returns, self.true_rewards_ph: true_rewards,
            self.learning_rate_ph: learning_rate, self.clip_range_ph: cliprange,
            self.old_neglog_pac_ph: neglogpacs, self.old_vpred_ph: values,
            self.observation_ph: obs, self.observation_next_ph: obs_next
        }
        if states is not None:
            td_map[self.train_model.states_ph] = states
            td_map[self.train_model.dones_ph] = masks

        if cliprange_vf is not None and cliprange_vf >= 0:
            td_map[self.clip_range_vf_ph] = cliprange_vf

        if states is None:
            update_fac = max(self.n_batch // self.nminibatches // self.noptepochs, 1)
        else:
            update_fac = max(self.n_batch // self.nminibatches // self.noptepochs // self.n_steps, 1)

        _ = self.sess.run([self.act_hat], {
            self.observation_ph: obs, self.observation_next_ph: obs_next,
        })

        _ = self.sess.run([self.obs_next_hat], {
            self.observation_ph: obs, self.action_ph: actions,
        })

        _ = self.sess.run([self.frw_loss], {
            self.observation_ph: obs, self.observation_next_ph: obs_next, self.action_ph: actions,
        })

        _ = self.sess.run([self.inv_loss], {
            self.observation_ph: obs, self.observation_next_ph: obs_next, self.action_ph: actions,
        })

        if writer is not None:
            # run loss backprop with summary, but once every 10 runs save the metadata (memory, compute time, ...)
            if self.full_tensorboard_log and (1 + update) % 10 == 0:
                run_options = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
                run_metadata = tf.compat.v1.RunMetadata()
                summary, policy_loss, value_loss, policy_entropy, approxkl, clipfrac, _ = self.sess.run(
                    [self.summary, self.pg_loss, self.vf_loss, self.entropy, self.approxkl, self.clipfrac, self._train],
                    td_map, options=run_options, run_metadata=run_metadata)
                writer.add_run_metadata(run_metadata, 'step%d' % (update * update_fac))
            else:
                summary, policy_loss, value_loss, int_loss, policy_entropy, approxkl, clipfrac, _ = self.sess.run(
                    [self.summary, self.pg_loss, self.vf_loss, self.int_loss, self.entropy, self.approxkl, self.clipfrac, self._train],
                    td_map)
            writer.add_summary(summary, (update * update_fac))
        else:
            policy_loss, value_loss, policy_entropy, approxkl, clipfrac, _ = self.sess.run(
                [self.pg_loss, self.vf_loss, self.entropy, self.approxkl, self.clipfrac, self._train], td_map)

        return policy_loss, value_loss, int_loss, policy_entropy, approxkl, clipfrac

    def _setup_learn(self):
        if self.env is None:
            raise ValueError("Error: cannot train the model without a valid environment, please set an environment with set_env(self, env) method.")
        if self.episode_reward is None:
            self.episode_reward = np.zeros((self.n_envs * self.n_runs,))
        if self.ep_info_buf is None:
            self.ep_info_buf = deque(maxlen=10 * self.n_envs * self.n_runs)

    def learn(self, total_timesteps, callback=None, log_interval=1, tb_log_name="PPO2", reset_num_timesteps=True):

        # Transform to callable if needed

        self.learning_rate = get_schedule_fn(self.learning_rate)
        self.cliprange = get_schedule_fn(self.cliprange)
        cliprange_vf = get_schedule_fn(self.cliprange_vf)

        new_tb_log = self._init_num_timesteps(reset_num_timesteps)
        callback = self._init_callback(callback)

        with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name, new_tb_log) as writer:
            self._setup_learn()

            t_first_start = time.time()
            n_updates = total_timesteps // self.n_batch

            callback.on_training_start(locals(), globals())

            for update in range(1, n_updates + 1):

                assert self.n_batch % self.nminibatches == 0, ("The number of minibatches (`nminibatches`) "
                                                               "is not a factor of the total number of samples "
                                                               "collected per rollout (`n_batch`), "
                                                               "some samples won't be used."
                                                               )
                batch_size = self.n_batch // self.nminibatches
                t_start = time.time()
                frac = 1.0 - (update - 1.0) / n_updates
                lr_now = self.learning_rate(frac)
                cliprange_now = self.cliprange(frac)
                cliprange_vf_now = cliprange_vf(frac)

                callback.on_rollout_start()
                # true_reward is the reward without discount
                rollout = self.runner.run(callback)

                # Unpack

                obs, obs_next, returns, masks, actions, values, neglogpacs, states, ep_infos, true_reward = rollout

                #for item in [obs, obs_next, returns, masks, actions, values, neglogpacs, states, true_reward]:
                #    if item is not None:
                #        print(item.shape)
                #print(ep_infos)

                callback.on_rollout_end()

                # Early stopping due to the callback
                if not self.runner.continue_training:
                    break

                self.ep_info_buf.extend(ep_infos)
                mb_loss_vals = []
                if states is None:  # nonrecurrent version
                    update_fac = max(self.n_batch // self.nminibatches // self.noptepochs, 1)
                    inds = np.arange(self.n_batch)
                    for epoch_num in range(self.noptepochs):
                        np.random.shuffle(inds)
                        for start in range(0, self.n_batch, batch_size):
                            timestep = self.num_timesteps // update_fac + ((epoch_num * self.n_batch + start) // batch_size)
                            end = start + batch_size
                            mbinds = inds[start:end]
                            slices = (arr[mbinds] for arr in (obs, obs_next, returns, true_reward, masks, actions, values, neglogpacs))
                            mb_loss_vals.append(self._train_step(lr_now, cliprange_now, *slices, writer=writer, update=timestep, cliprange_vf=cliprange_vf_now))
                else:  # recurrent version
                    update_fac = max(self.n_batch // self.nminibatches // self.noptepochs // self.n_steps, 1)
                    assert self.n_envs % self.nminibatches == 0
                    env_indices = np.arange(self.n_envs)
                    flat_indices = np.arange(self.n_envs * self.n_steps).reshape(self.n_envs, self.n_steps)
                    envs_per_batch = batch_size // self.n_steps
                    for epoch_num in range(self.noptepochs):
                        np.random.shuffle(env_indices)
                        for start in range(0, self.n_envs, envs_per_batch):
                            timestep = self.num_timesteps // update_fac + ((epoch_num * self.n_envs + start) // envs_per_batch)
                            end = start + envs_per_batch
                            mb_env_inds = env_indices[start:end]
                            mb_flat_inds = flat_indices[mb_env_inds].ravel()
                            slices = (arr[mb_flat_inds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                            mb_states = states[mb_env_inds]
                            mb_loss_vals.append(self._train_step(lr_now, cliprange_now, *slices, update=timestep,
                                                                 writer=writer, states=mb_states,
                                                                 cliprange_vf=cliprange_vf_now))

                loss_vals = np.mean(mb_loss_vals, axis=0)
                t_now = time.time()
                fps = int(self.n_batch / (t_now - t_start))

                if writer is not None:
                    total_episode_reward_logger(self.episode_reward,
                                                true_reward.reshape((self.n_envs * self.n_runs, self.n_steps)),
                                                masks.reshape((self.n_envs * self.n_runs, self.n_steps)),
                                                writer, self.num_timesteps)

                if self.verbose >= 1 and (update % log_interval == 0 or update == 1):
                    explained_var = explained_variance(values, returns)
                    logger.logkv("serial_timesteps", update * self.n_steps)
                    logger.logkv("n_updates", update)
                    logger.logkv("total_timesteps", self.num_timesteps)
                    logger.logkv("fps", fps)
                    logger.logkv("explained_variance", float(explained_var))
                    if len(self.ep_info_buf) > 0 and len(self.ep_info_buf[0]) > 0:
                        logger.logkv('ep_reward_mean', safe_mean([ep_info['r'] for ep_info in self.ep_info_buf]))
                        logger.logkv('ep_normal_mean', safe_mean([ep_info['n'] for ep_info in self.ep_info_buf]))
                        logger.logkv('ep_attack_mean', safe_mean([ep_info['a'] for ep_info in self.ep_info_buf]))
                        logger.logkv('ep_precision_mean', safe_mean([ep_info['p'] for ep_info in self.ep_info_buf]))
                    logger.logkv('time_elapsed', t_start - t_first_start)
                    for (loss_val, loss_name) in zip(loss_vals, self.loss_names):
                        logger.logkv(loss_name, loss_val)
                    logger.dumpkvs()

            callback.on_training_end()
            return self

    def demo(self, ntests=10):
        assert self.env.num_envs == 1, "You must pass only one environment when using this function"
        normal_passed = []
        attack_blocked = []
        ids_precision = []
        episode_reward = []
        for episode in range(ntests):
            rollout = self.runner._run()
            #obs, returns, masks, actions, values, neglogpacs, states, ep_infos, true_reward = rollout
            obs, obs_next, returns, masks, actions, values, neglogpacs, states, ep_infos, true_reward = rollout
            normal_passed.append(ep_infos[0]['n'])
            attack_blocked.append(ep_infos[0]['a'])
            ids_precision.append(ep_infos[0]['p'])
            episode_reward.append(ep_infos[0]['r'])
            print(ep_infos[0]['r'])
        print(f'Normal traffic passed: {np.mean(normal_passed)}')
        print(f'Malicious traffic blocked: {np.mean(attack_blocked)}')
        print(f'IDS precision: {np.mean(ids_precision)}')
        print(f'Episode reward: {np.mean(episode_reward)}')
        return episode_reward, normal_passed, attack_blocked, ids_precision

    def save(self, save_path, cloudpickle=False):
        data = {
            "gamma": self.gamma,
            "n_steps": self.n_steps,
            "vf_coef": self.vf_coef,
            "ent_coef": self.ent_coef,
            "max_grad_norm": self.max_grad_norm,
            "learning_rate": self.learning_rate,
            "lam": self.lam,
            "nminibatches": self.nminibatches,
            "noptepochs": self.noptepochs,
            "cliprange": self.cliprange,
            "cliprange_vf": self.cliprange_vf,
            "verbose": self.verbose,
            "policy": self.policy,
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "n_envs": self.n_envs,
            "n_runs": self.n_runs,
            "n_cpu_tf_sess": self.n_cpu_tf_sess,
            "seed": self.seed,
            "_vectorize_action": self._vectorize_action,
            "policy_kwargs": self.policy_kwargs
        }

        params_to_save = self.get_parameters()

        self._save_to_file(save_path, data=data, params=params_to_save, cloudpickle=cloudpickle)


class Runner(AbstractEnvRunner):

    def __init__(self, *, env, model, n_runs, n_steps, gamma, lam):
        """
        A runner to learn the policy of an environment for a model

        :param env: (Gym environment) The environment to learn from
        :param model: (Model) The model to learn
        :param n_steps: (int) The number of steps to run for each environment
        :param gamma: (float) Discount factor
        :param lam: (float) Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        """
        super().__init__(env=env, model=model, n_steps=n_steps)
        self.lam = lam
        self.gamma = gamma

        self.mb_obs = [[] for _ in range(self.n_envs)]
        self.mb_obs_next = [[] for _ in range(self.n_envs)]
        self.mb_actions = [[] for _ in range(self.n_envs)]
        self.mb_values = [[] for _ in range(self.n_envs)]
        self.mb_neglogpacs = [[] for _ in range(self.n_envs)]
        self.mb_dones = [[] for _ in range(self.n_envs)]
        self.mb_rewards = [[] for _ in range(self.n_envs)]
        self.scores = [[] for _ in range(self.n_envs)]

        self.n_runs = n_runs

        self.obs = np.zeros((env.num_envs * self.n_runs,) + env.observation_space.shape, dtype=env.observation_space.dtype.name)
        self.dones = [False for _ in range(env.num_envs * self.n_runs)]

    def _run_one(self, env_idx, run_idx):

        tstart = time.time()

        for _ in range(self.n_steps):

            # step model

            actions, values, self.states, neglogpacs = self.model.step(self.obs[run_idx:run_idx + 1], self.states, self.dones[run_idx:run_idx + 1])

            last_obs = self.obs[run_idx:run_idx + 1].copy()

            # save results

            self.mb_obs[run_idx].append(self.obs.copy()[run_idx])
            self.mb_actions[run_idx].append(actions[0])
            self.mb_values[run_idx].append(values[0])
            self.mb_neglogpacs[run_idx].append(neglogpacs[0])
            self.mb_dones[run_idx].append(self.dones[run_idx])

            # Clip the actions to avoid out of bound error

            clipped_actions = actions
            if isinstance(self.env.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.env.action_space.low, self.env.action_space.high)

            tnow = time.time()

            self.obs[run_idx], reward, self.dones[run_idx], infos = self.env.step_one(env_idx, clipped_actions)

            int_reward = self.model.intrinsic_reward(last_obs, self.obs[run_idx:run_idx + 1], actions)

            self.mb_obs_next[run_idx].append(self.obs.copy()[run_idx])

            self.mb_rewards[run_idx].append(reward + int_reward)
            self.scores[run_idx].append([reward, infos['n'], infos['a'], infos['p']])

            self.model.num_timesteps += 1

            if self.callback is not None:

                # Abort training early

                self.callback.update_locals(locals())
                if self.callback.on_step() is False:
                    self.continue_training = False

                    # Return dummy values

                    return [None] * 9

            #self.mb_rewards[env_idx].append(rewards)

        print(f'Step time in {env_idx}: {(time.time() - tstart) / self.n_steps}')

        #self.mb_rewards[env_idx], infos = self.env.reward_one(env_idx)
        #self.scores[env_idx] = [[info['r'], info['n'], info['a'], info['p']] for info in infos]

    def _run(self):
        """
        Run a learning step of the model

        :return:
            - observations: (np.ndarray) the observations
            - rewards: (np.ndarray) the rewards
            - masks: (numpy bool) whether an episode is over or not
            - actions: (np.ndarray) the actions
            - values: (np.ndarray) the value function output
            - negative log probabilities: (np.ndarray)
            - states: (np.ndarray) the internal states of the recurrent policies
            - infos: (dict) the extra information of the model
        """
        # mb stands for minibatch

        self.mb_obs = [[] for _ in range(self.n_envs * self.n_runs)]
        self.mb_obs_next = [[] for _ in range(self.n_envs * self.n_runs)]
        self.mb_actions = [[] for _ in range(self.n_envs * self.n_runs)]
        self.mb_values = [[] for _ in range(self.n_envs * self.n_runs)]
        self.mb_neglogpacs = [[] for _ in range(self.n_envs * self.n_runs)]
        self.mb_dones = [[] for _ in range(self.n_envs * self.n_runs)]
        self.mb_rewards = [[] for _ in range(self.n_envs * self.n_runs)]
        self.scores = [[] for _ in range(self.n_envs * self.n_runs)]

        ep_infos = []

        for i in range(self.n_runs):
            self.obs[i * self.n_envs : (i + 1) * self.n_envs] = self.env.reset()

            # run steps in different threads

            threads = []
            for env_idx in range(self.n_envs):
                th = Thread(target=self._run_one, args=(env_idx, i * self.n_envs + env_idx))
                th.start()
                threads.append(th)
            for th in threads:
                th.join()

        # combine data gathered into batches

        mb_obs = [np.stack([self.mb_obs[idx][step] for idx in range(self.n_envs * self.n_runs)]) for step in range(self.n_steps)]
        mb_obs_next = [np.stack([self.mb_obs_next[idx][step] for idx in range(self.n_envs * self.n_runs)]) for step in range(self.n_steps)]
        mb_rewards = [np.hstack([self.mb_rewards[idx][step] for idx in range(self.n_envs * self.n_runs)]) for step in range(self.n_steps)]
        mb_actions = [np.hstack([self.mb_actions[idx][step] for idx in range(self.n_envs * self.n_runs)]) for step in range(self.n_steps)]
        mb_values = [np.hstack([self.mb_values[idx][step] for idx in range(self.n_envs * self.n_runs)]) for step in range(self.n_steps)]
        mb_neglogpacs = [np.hstack([self.mb_neglogpacs[idx][step] for idx in range(self.n_envs * self.n_runs)]) for step in range(self.n_steps)]
        mb_dones = [np.hstack([self.mb_dones[idx][step] for idx in range(self.n_envs * self.n_runs)]) for step in range(self.n_steps)]
        mb_scores = [np.vstack([self.scores[idx][step] for step in range(self.n_steps)]) for idx in range(self.n_envs * self.n_runs)]
        mb_states = self.states
        self.dones = np.array(self.dones)

        for scores_in_env in mb_scores:
            maybe_ep_info = {
                'r': safe_mean(scores_in_env[:, 0]),
                'n': safe_mean(scores_in_env[:, 1]),
                'a': safe_mean(scores_in_env[:, 2]),
                'p': safe_mean(scores_in_env[:, 3])
            }
            ep_infos.append(maybe_ep_info)

        # batch of steps to batch of rollouts

        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_obs_next = np.asarray(mb_obs_next, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=bool)
        last_values = self.model.value(self.obs, self.states, self.dones)

        # discount/bootstrap off value fn

        mb_advs = np.zeros_like(mb_rewards)
        true_reward = np.copy(mb_rewards)
        last_gae_lam = 0
        for step in reversed(range(self.n_steps)):
            if step == self.n_steps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[step + 1]
                nextvalues = mb_values[step + 1]
            delta = mb_rewards[step] + self.gamma * nextvalues * nextnonterminal - mb_values[step]
            mb_advs[step] = last_gae_lam = delta + self.gamma * self.lam * nextnonterminal * last_gae_lam
        mb_returns = mb_advs + mb_values

        mb_obs, mb_obs_next, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, true_reward = map(
            swap_and_flatten, (mb_obs, mb_obs_next, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, true_reward)
        )

        # reset data

        self.mb_obs = [[] for _ in range(self.n_envs)]
        self.mb_obs_next = [[] for _ in range(self.n_envs)]
        self.mb_actions = [[] for _ in range(self.n_envs)]
        self.mb_values = [[] for _ in range(self.n_envs)]
        self.mb_neglogpacs = [[] for _ in range(self.n_envs)]
        self.mb_dones = [[] for _ in range(self.n_envs)]
        self.mb_rewards = [[] for _ in range(self.n_envs)]
        self.scores = [[] for _ in range(self.n_envs)]

        return mb_obs, mb_obs_next, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, mb_states, ep_infos, true_reward

    def _run_(self):
        """
        Run a learning step of the model

        :return:
            - observations: (np.ndarray) the observations
            - rewards: (np.ndarray) the rewards
            - masks: (numpy bool) whether an episode is over or not
            - actions: (np.ndarray) the actions
            - values: (np.ndarray) the value function output
            - negative log probabilities: (np.ndarray)
            - states: (np.ndarray) the internal states of the recurrent policies
            - infos: (dict) the extra information of the model
        """
        # mb stands for minibatch

        self.obs[:] = self.env.reset()

        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [], [], [], [], [], [], []
        mb_states = self.states
        ep_infos = []
        scores = [[] for _ in range(self.n_envs)]
        normals = [[] for _ in range(self.n_envs)]
        attacks = [[] for _ in range(self.n_envs)]
        precisions = [[] for _ in range(self.n_envs)]
        telapsed = 0
        for _ in range(self.n_steps):
            actions, values, self.states, neglogpacs = self.model.step(self.obs, self.states, self.dones)
            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.env.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.env.action_space.low, self.env.action_space.high)
            tnow = time.time()
            self.obs[:], rewards, self.dones, infos = self.env.step(clipped_actions)
            telapsed += time.time() - tnow
            for ri in range(self.n_envs):
                scores[ri].append(infos[ri]['r'])
                normals[ri].append(infos[ri]['n'])
                attacks[ri].append(infos[ri]['a'])
                precisions[ri].append(infos[ri]['p'])

            self.model.num_timesteps += self.n_envs

            if self.callback is not None:
                # Abort training early
                self.callback.update_locals(locals())
                if self.callback.on_step() is False:
                    self.continue_training = False
                    # Return dummy values
                    return [None] * 9
            mb_rewards.append(rewards)

        print('Time step: {0}'.format(telapsed / self.n_steps))

        for s, n, a, p in zip(scores, normals, attacks, precisions):
            maybe_ep_info = {'r': safe_mean(s), 'n': safe_mean(n), 'a': safe_mean(a), 'p': safe_mean(p)}
            if maybe_ep_info is not None:
                ep_infos.append(maybe_ep_info)

        # batch of steps to batch of rollouts

        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.model.value(self.obs, self.states, self.dones)

        # discount/bootstrap off value fn

        mb_advs = np.zeros_like(mb_rewards)
        true_reward = np.copy(mb_rewards)
        last_gae_lam = 0
        for step in reversed(range(self.n_steps)):
            if step == self.n_steps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[step + 1]
                nextvalues = mb_values[step + 1]
            delta = mb_rewards[step] + self.gamma * nextvalues * nextnonterminal - mb_values[step]
            mb_advs[step] = last_gae_lam = delta + self.gamma * self.lam * nextnonterminal * last_gae_lam
        mb_returns = mb_advs + mb_values

        mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, true_reward = map(
            swap_and_flatten, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, true_reward)
        )

        return mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, mb_states, ep_infos, true_reward


def swap_and_flatten(arr):
    shape = arr.shape
    return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])
