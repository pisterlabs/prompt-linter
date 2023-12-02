import ray
import time
import gym
import numpy as np
import tensorflow as tf
import pandas as pd

from collections import deque
from stable_baselines import logger
from stable_baselines.common import explained_variance, ActorCriticRLModel, tf_util, SetVerbosity, TensorboardWriter
from stable_baselines.common.runners import AbstractEnvRunner
from stable_baselines.common.policies import ActorCriticPolicy, RecurrentActorCriticPolicy
from stable_baselines.a2c.utils import total_episode_reward_logger
from latent_gce.trajectory_utils import mp_trajectories_to_input, \
    mp_collect_input_from_state, mp_mj_collect_input_from_state, mp_mj_collect_input_from_state_return_final
from latent_gce.model import LatentGCEImage, LatentGCEIdentity


class GcePPO(ActorCriticRLModel):
    """
    Unsupervised learning using empowerment from Latent-GCE using PPO.
    Adapted from Stable-baseline's PPO implementation.
    """
    def __init__(self, exp_name, policy, env, emp_trajectory_options, emp_options, gamma=0.99, n_steps=128,
                 ent_coef=0.01, learning_rate=2.5e-4, vf_coef=0.5, max_grad_norm=0.5, lam=0.95, nminibatches=4,
                 noptepochs=4, cliprange=0.2, cliprange_vf=None,
                 verbose=0, tensorboard_log=None, _init_setup_model=True, policy_kwargs=None,
                 full_tensorboard_log=False, seed=None, n_cpu_tf_sess=None, mode='identity'):

        super(GcePPO, self).__init__(policy=policy, env=env, verbose=verbose, requires_vec_env=True,
                                     _init_setup_model=_init_setup_model, policy_kwargs=policy_kwargs,
                                     seed=seed, n_cpu_tf_sess=n_cpu_tf_sess)

        self.emp_trajectory_options = emp_trajectory_options
        self.emp_options = emp_options
        self.exp_name = exp_name
        self.timesteps_array = []
        self.episode_reward_array = []
        self.episode_length_array = []

        self.emp_logging_keys = self.emp_options.get('logging')
        self.emp_logging_arrays = {}
        if self.emp_logging_keys:
            for k in self.emp_logging_keys:
                self.emp_logging_arrays[k] = []

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

        self.graph = None
        self.sess = None
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
        self.params = None
        self._train = None
        self.loss_names = None
        self.train_model = None
        self.act_model = None
        self.step = None
        self.proba_step = None
        self.value = None
        self.initial_state = None
        self.n_batch = None
        self.summary = None
        self.episode_reward = None

        self.runner = None
        self.obs = None
        self.no_reset_at_all = False
        self.mode = mode

        if _init_setup_model:
            self.setup_model()

    def _get_pretrain_placeholders(self):
        policy = self.act_model
        if isinstance(self.action_space, gym.spaces.Discrete):
            return policy.obs_ph, self.action_ph, policy.policy
        return policy.obs_ph, self.action_ph, policy.deterministic_action

    def setup_model(self):
        with SetVerbosity(self.verbose):

            assert issubclass(self.policy, ActorCriticPolicy), "Error: the input policy for the PPO2 model must be " \
                                                               "an instance of common.policies.ActorCriticPolicy."

            self.n_batch = self.n_envs * self.n_steps

            self.graph = tf.Graph()
            with self.graph.as_default():
                self.set_random_seed(self.seed)
                self.sess = tf_util.make_session(num_cpu=self.n_cpu_tf_sess, graph=self.graph)

                n_batch_step = None
                n_batch_train = None
                if issubclass(self.policy, RecurrentActorCriticPolicy):
                    assert self.n_envs % self.nminibatches == 0, \
                        "For recurrent policies, " \
                        "the number of environments run in parallel should be a multiple of nminibatches."
                    n_batch_step = self.n_envs
                    n_batch_train = self.n_batch // self.nminibatches

                act_model = self.policy(self.sess, self.observation_space, self.action_space, self.n_envs, 1,
                                        n_batch_step, reuse=False, **self.policy_kwargs)
                with tf.variable_scope("train_model", reuse=True,
                                       custom_getter=tf_util.outer_scope_getter("train_model")):
                    train_model = self.policy(self.sess, self.observation_space, self.action_space,
                                              self.n_envs // self.nminibatches, self.n_steps, n_batch_train,
                                              reuse=True, **self.policy_kwargs)

                with tf.variable_scope("loss", reuse=False):
                    self.action_ph = train_model.pdtype.sample_placeholder([None], name="action_ph")
                    self.advs_ph = tf.placeholder(tf.float32, [None], name="advs_ph")
                    self.rewards_ph = tf.placeholder(tf.float32, [None], name="rewards_ph")
                    self.old_neglog_pac_ph = tf.placeholder(tf.float32, [None], name="old_neglog_pac_ph")
                    self.old_vpred_ph = tf.placeholder(tf.float32, [None], name="old_vpred_ph")
                    self.learning_rate_ph = tf.placeholder(tf.float32, [], name="learning_rate_ph")
                    self.clip_range_ph = tf.placeholder(tf.float32, [], name="clip_range_ph")

                    neglogpac = train_model.proba_distribution.neglogp(self.action_ph)
                    self.entropy = tf.reduce_mean(train_model.proba_distribution.entropy())

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
                        self.clip_range_vf_ph = tf.placeholder(tf.float32, [], name="clip_range_vf_ph")

                    if self.clip_range_vf_ph is None:
                        # No clipping
                        vpred_clipped = train_model.value_flat
                    else:
                        # Clip the different between old and new value
                        # NOTE: this depends on the reward scaling
                        vpred_clipped = self.old_vpred_ph + \
                                        tf.clip_by_value(train_model.value_flat - self.old_vpred_ph,
                                                         - self.clip_range_vf_ph, self.clip_range_vf_ph)

                    vf_losses1 = tf.square(vpred - self.rewards_ph)
                    vf_losses2 = tf.square(vpred_clipped - self.rewards_ph)
                    self.vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))

                    ratio = tf.exp(self.old_neglog_pac_ph - neglogpac)
                    pg_losses = -self.advs_ph * ratio
                    pg_losses2 = -self.advs_ph * tf.clip_by_value(ratio, 1.0 - self.clip_range_ph, 1.0 +
                                                                  self.clip_range_ph)
                    self.pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
                    self.approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - self.old_neglog_pac_ph))
                    self.clipfrac = tf.reduce_mean(tf.cast(tf.greater(tf.abs(ratio - 1.0),
                                                                      self.clip_range_ph), tf.float32))
                    loss = self.pg_loss - self.entropy * self.ent_coef + self.vf_loss * self.vf_coef

                    tf.summary.scalar('entropy_loss', self.entropy)
                    tf.summary.scalar('policy_gradient_loss', self.pg_loss)
                    tf.summary.scalar('value_function_loss', self.vf_loss)
                    tf.summary.scalar('approximate_kullback-leibler', self.approxkl)
                    tf.summary.scalar('clip_factor', self.clipfrac)
                    tf.summary.scalar('loss', loss)

                    with tf.variable_scope('model'):
                        self.params = tf.trainable_variables()
                        if self.full_tensorboard_log:
                            for var in self.params:
                                tf.summary.histogram(var.name, var)
                    grads = tf.gradients(loss, self.params)
                    if self.max_grad_norm is not None:
                        grads, _grad_norm = tf.clip_by_global_norm(grads, self.max_grad_norm)
                    grads = list(zip(grads, self.params))
                trainer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph, epsilon=1e-5)
                self._train = trainer.apply_gradients(grads)

                self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac']

                with tf.variable_scope("input_info", reuse=False):
                    tf.summary.scalar('discounted_rewards', tf.reduce_mean(self.rewards_ph))
                    tf.summary.scalar('learning_rate', tf.reduce_mean(self.learning_rate_ph))
                    tf.summary.scalar('advantage', tf.reduce_mean(self.advs_ph))
                    tf.summary.scalar('clip_range', tf.reduce_mean(self.clip_range_ph))
                    if self.clip_range_vf_ph is not None:
                        tf.summary.scalar('clip_range_vf', tf.reduce_mean(self.clip_range_vf_ph))

                    tf.summary.scalar('old_neglog_action_probability', tf.reduce_mean(self.old_neglog_pac_ph))
                    tf.summary.scalar('old_value_pred', tf.reduce_mean(self.old_vpred_ph))

                    if self.full_tensorboard_log:
                        tf.summary.histogram('discounted_rewards', self.rewards_ph)
                        tf.summary.histogram('learning_rate', self.learning_rate_ph)
                        tf.summary.histogram('advantage', self.advs_ph)
                        tf.summary.histogram('clip_range', self.clip_range_ph)
                        tf.summary.histogram('old_neglog_action_probability', self.old_neglog_pac_ph)
                        tf.summary.histogram('old_value_pred', self.old_vpred_ph)
                        if tf_util.is_image(self.observation_space):
                            tf.summary.image('observation', train_model.obs_ph)
                        else:
                            tf.summary.histogram('observation', train_model.obs_ph)

                self.train_model = train_model
                self.act_model = act_model
                self.step = act_model.step
                self.proba_step = act_model.proba_step
                self.value = act_model.value
                self.initial_state = act_model.initial_state
                tf.global_variables_initializer().run(session=self.sess)  # pylint: disable=E1101

                self.summary = tf.summary.merge_all()

    def _train_step(self, learning_rate, cliprange, obs, returns, masks, actions, values, neglogpacs, update,
                    writer, states=None, cliprange_vf=None):
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
        td_map = {self.train_model.obs_ph: obs, self.action_ph: actions,
                  self.advs_ph: advs, self.rewards_ph: returns,
                  self.learning_rate_ph: learning_rate, self.clip_range_ph: cliprange,
                  self.old_neglog_pac_ph: neglogpacs, self.old_vpred_ph: values}
        if states is not None:
            td_map[self.train_model.states_ph] = states
            td_map[self.train_model.dones_ph] = masks

        if cliprange_vf is not None and cliprange_vf >= 0:
            td_map[self.clip_range_vf_ph] = cliprange_vf

        if states is None:
            update_fac = self.n_batch // self.nminibatches // self.noptepochs + 1
        else:
            update_fac = self.n_batch // self.nminibatches // self.noptepochs // self.n_steps + 1

        if writer is not None:
            # run loss backprop with summary, but once every 10 runs save the metadata (memory, compute time, ...)
            if self.full_tensorboard_log and (1 + update) % 10 == 0:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, policy_loss, value_loss, policy_entropy, approxkl, clipfrac, _ = self.sess.run(
                    [self.summary, self.pg_loss, self.vf_loss, self.entropy, self.approxkl, self.clipfrac, self._train],
                    td_map, options=run_options, run_metadata=run_metadata)
                writer.add_run_metadata(run_metadata, 'step%d' % (update * update_fac))
            else:
                summary, policy_loss, value_loss, policy_entropy, approxkl, clipfrac, _ = self.sess.run(
                    [self.summary, self.pg_loss, self.vf_loss, self.entropy, self.approxkl, self.clipfrac, self._train],
                    td_map)
            writer.add_summary(summary, (update * update_fac))
        else:
            policy_loss, value_loss, policy_entropy, approxkl, clipfrac, _ = self.sess.run(
                [self.pg_loss, self.vf_loss, self.entropy, self.approxkl, self.clipfrac, self._train], td_map)

        return policy_loss, value_loss, policy_entropy, approxkl, clipfrac

    def learn(self, total_timesteps, callback=None, log_interval=1, tb_log_name="GCE-PPO",
              reset_num_timesteps=True, dump_log=True):
        # Transform to callable if needed
        self.learning_rate = get_schedule_fn(self.learning_rate)
        self.cliprange = get_schedule_fn(self.cliprange)
        cliprange_vf = get_schedule_fn(self.cliprange_vf)

        new_tb_log = self._init_num_timesteps(reset_num_timesteps)

        with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name, new_tb_log) \
                as writer:
            self._setup_learn()

            if not self.runner:
                self.runner = Runner(env=self.env, model=self, n_steps=self.n_steps, gamma=self.gamma, lam=self.lam,
                                     emp_trajectory_options=self.emp_trajectory_options, emp_options=self.emp_options,
                                     mode=self.mode, tensorboard_log=self.tensorboard_log)
            runner = self.runner
            self.episode_reward = np.zeros((self.n_envs,))

            ep_info_buf = deque(maxlen=100)
            t_first_start = time.time()

            n_updates = total_timesteps // self.n_batch
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
                # true_reward is the reward without discount
                obs, returns, masks, actions, values, neglogpacs, states, ep_infos, true_reward = runner.run()
                self.obs = obs
                self.num_timesteps += self.n_batch
                ep_info_buf.extend(ep_infos)
                mb_loss_vals = []
                if states is None:  # nonrecurrent version
                    update_fac = self.n_batch // self.nminibatches // self.noptepochs + 1
                    inds = np.arange(self.n_batch)
                    for epoch_num in range(self.noptepochs):
                        np.random.shuffle(inds)
                        for start in range(0, self.n_batch, batch_size):
                            timestep = self.num_timesteps // update_fac + ((self.noptepochs * self.n_batch + epoch_num *
                                                                            self.n_batch + start) // batch_size)
                            end = start + batch_size
                            mbinds = inds[start:end]
                            slices = (arr[mbinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                            mb_loss_vals.append(self._train_step(lr_now, cliprange_now, *slices, writer=writer,
                                                                 update=timestep, cliprange_vf=cliprange_vf_now))
                else:  # recurrent version
                    update_fac = self.n_batch // self.nminibatches // self.noptepochs // self.n_steps + 1
                    assert self.n_envs % self.nminibatches == 0
                    env_indices = np.arange(self.n_envs)
                    flat_indices = np.arange(self.n_envs * self.n_steps).reshape(self.n_envs, self.n_steps)
                    envs_per_batch = batch_size // self.n_steps
                    for epoch_num in range(self.noptepochs):
                        np.random.shuffle(env_indices)
                        for start in range(0, self.n_envs, envs_per_batch):
                            timestep = self.num_timesteps // update_fac + ((self.noptepochs * self.n_envs + epoch_num *
                                                                            self.n_envs + start) // envs_per_batch)
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
                                                true_reward.reshape((self.n_envs, self.n_steps)),
                                                masks.reshape((self.n_envs, self.n_steps)),
                                                writer, self.num_timesteps)

                if self.verbose >= 1 and (update % log_interval == 0 or update == 1):
                    explained_var = explained_variance(values, returns)
                    logger.logkv("serial_timesteps", update * self.n_steps)
                    logger.logkv("n_updates", update)
                    logger.logkv("total_timesteps", self.num_timesteps)
                    logger.logkv("fps", fps)
                    logger.logkv("explained_variance", float(explained_var))
                    if len(ep_info_buf) > 0 and len(ep_info_buf[0]) > 0:
                        logger.logkv('ep_reward_mean', safe_mean([ep_info['r'] for ep_info in ep_info_buf]))
                        logger.logkv('ep_len_mean', safe_mean([ep_info['l'] for ep_info in ep_info_buf]))
                        self.timesteps_array.append(self.num_timesteps)
                        self.episode_reward_array.append(safe_mean([ep_info['r'] for ep_info in ep_info_buf]))
                        self.episode_length_array.append(safe_mean([ep_info['l'] for ep_info in ep_info_buf]))

                        if self.emp_logging_keys:
                            for k in self.emp_logging_keys:
                                logger.logkv(k, safe_mean([ep_info[k] for ep_info in ep_info_buf]))
                                self.emp_logging_arrays[k].append(safe_mean([ep_info[k] for ep_info in ep_info_buf]))

                        ep_info_buf.clear()

                    logger.logkv('time_elapsed', t_start - t_first_start)
                    for (loss_val, loss_name) in zip(loss_vals, self.loss_names):
                        logger.logkv(loss_name, loss_val)
                    logger.dumpkvs()

                if callback is not None:
                    # Only stop training if return value is False, not when it is None. This is for backwards
                    # compatibility with callbacks that have no return statement.
                    if callback(locals(), globals()) is False:
                        break

            if dump_log:
                logging_dict = {'Steps': self.timesteps_array,
                                'Episode Reward': self.episode_reward_array,
                                'Episode Length': self.episode_length_array}
                if self.emp_logging_keys:
                    for k in self.emp_logging_keys:
                        logging_dict[k] = self.emp_logging_arrays[k]
                df = pd.DataFrame(logging_dict)
                df.to_csv(self.tensorboard_log + '/' + self.exp_name + '.csv', index=False)

            save_emp_model = self.emp_options.get('save_model')
            if save_emp_model:
                self.runner.gce_model.save(save_emp_model)

            return self

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
            "n_cpu_tf_sess": self.n_cpu_tf_sess,
            "seed": self.seed,
            "_vectorize_action": self._vectorize_action,
            "policy_kwargs": self.policy_kwargs
        }

        params_to_save = self.get_parameters()

        self._save_to_file(save_path, data=data, params=params_to_save, cloudpickle=cloudpickle)


class Runner(AbstractEnvRunner):
    def __init__(self, *, env, model, n_steps, gamma, lam, emp_trajectory_options, emp_options, mode,
                 tensorboard_log):
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

        self.emp_trajectory_options = emp_trajectory_options
        self.emp_history = None
        self.reward_weight = emp_options['reward_weight']
        self.emp_weight = emp_options['emp_weight']
        self.emp_history_size = emp_options['buffer_size']
        self.emp_obs_selection = emp_options['obs_selection']
        self.is_mujoco = emp_options['is_mujoco']
        self.action_penalty = emp_options['action_penalty']

        self.exp_emp = emp_options.get('exp_emp')
        self.logging = emp_options.get('logging')
        self.uniform_actions = emp_options.get('uniform_actions')
        self.multiplicative_emp = emp_options.get('multiplicative')

        lr = emp_options['learning_rate']
        obs_raw_dim = env.observation_space.shape[0] * emp_trajectory_options['num_steps_observation']
        if mode == 'identity':
            self.gce_model = LatentGCEIdentity(obs_raw_dim=obs_raw_dim,
                                               action_raw_dim=env.action_space.shape[0] * emp_trajectory_options['T'],
                                               obs_selection=self.emp_obs_selection,
                                               learning_rate=lr,
                                               log_dir=tensorboard_log)
        elif mode == 'pixels':
            self.gce_model = LatentGCEImage(env=env,
                                            num_steps_observation=2,
                                            action_raw_dim=env.action_space.shape[0] * emp_trajectory_options['T'],
                                            learning_rate=lr,
                                            state_latent_dimension=32,
                                            action_latent_dimension=32,
                                            log_dir=tensorboard_log)

        if self.emp_weight != 0:
            ray.init(num_cpus=16)
        self.gce_train_loss = 0
        self.total_random_steps = 0

    def run(self):
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
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [], [], [], [], [], []
        mb_states = self.states
        ep_infos = []

        final_trajectories_obs = []
        final_trajectories_actions = []
        final_trajectories_neg_log_prob = []
        trajectories_obs = []
        trajectories_actions = []
        trajectories_neg_log_prob = []

        # Save mujoco states for resetting
        mb_mujoco_sim = []

        for o in self.obs:
            trajectories_obs.append([o])
            trajectories_actions.append([])
            trajectories_neg_log_prob.append([])

        for _ in range(self.n_steps):
            actions, values, self.states, neglogpacs = self.model.step(self.obs, self.states, self.dones)

            # Mujoco
            if self.is_mujoco and self.uniform_actions:
                for e in self.env.envs:
                    if np.random.rand() < 1 / self.emp_trajectory_options['total_steps']:
                        qpos = e.env.unwrapped.sim.data.qpos.copy()
                        qvel = e.env.unwrapped.sim.data.qvel.copy()
                        mb_mujoco_sim.append(np.array([qpos, qvel]))

            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.env.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.env.action_space.low, self.env.action_space.high)
            self.obs[:], rewards, self.dones, infos = self.env.step(clipped_actions)
            for idx in range(len(infos)):
                maybe_ep_info = infos[idx].get('episode')
                if maybe_ep_info is not None:
                    if self.logging:
                        for k in self.logging:
                            maybe_ep_info[k] = infos[idx][k]
                    ep_infos.append(maybe_ep_info)
                maybe_terminal_observation = infos[idx].get('terminal_observation')

                if self.emp_weight != 0 and not self.uniform_actions:
                    trajectories_actions[idx].append(np.copy(actions[idx]))
                    trajectories_neg_log_prob[idx].append(np.copy(neglogpacs[idx]))
                    if maybe_terminal_observation is None:
                        trajectories_obs[idx].append(np.copy(self.obs[idx]))
                    else:
                        trajectories_obs[idx].append(np.copy(maybe_terminal_observation))
                        final_trajectories_obs.append(trajectories_obs[idx])
                        final_trajectories_actions.append(trajectories_actions[idx])
                        final_trajectories_neg_log_prob.append(trajectories_neg_log_prob[idx])
                        trajectories_obs[idx] = [np.copy(self.obs[idx])]
                        trajectories_actions[idx] = []
                        trajectories_neg_log_prob[idx] = []

            mb_rewards.append(rewards)

        # batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)

        if self.emp_weight != 0:
            if not self.uniform_actions:
                final_trajectories_obs += trajectories_obs
                final_trajectories_actions += trajectories_actions
                final_trajectories_neg_log_prob += trajectories_neg_log_prob
                emp_data = mp_trajectories_to_input((final_trajectories_obs, final_trajectories_actions),
                                                    self.emp_trajectory_options,
                                                    neg_log_prob=final_trajectories_neg_log_prob)
                self.update_emp_history(emp_data)
            elif self.is_mujoco:
                self.collect_uniform_emp_data(mb_mujoco_sim)
            else:
                self.collect_uniform_emp_data(mb_obs)

        if self.emp_weight != 0:
            print('EMP Begin...')
            single_obs = mb_obs.reshape(([mb_obs.shape[0] * mb_obs.shape[1]] + list(mb_obs.shape[2:])))
            if self.emp_trajectory_options['num_steps_observation'] == 1:
                emp_obs = single_obs
            elif self.emp_trajectory_options['num_steps_observation'] == 2:
                emp_obs = np.concatenate([single_obs[:-1], single_obs[1:]], axis=1)
            else:
                emp_obs = None
            emp_rewards = self.gce_model.water_filling_from_observations(emp_obs, batch_size=1024)
            if self.emp_trajectory_options['num_steps_observation'] == 2:
                emp_rewards = np.insert(emp_rewards, len(emp_rewards), 0)
            if self.exp_emp:
                emp_rewards = np.exp(emp_rewards)
            if self.multiplicative_emp:
                mb_rewards = self.multiplicative_emp * mb_rewards * emp_rewards.reshape(mb_rewards.shape)
            else:
                mb_rewards = self.reward_weight * mb_rewards + self.emp_weight * emp_rewards.reshape(mb_rewards.shape)
            # Action penalty
            if self.action_penalty:
                mb_rewards -= self.action_penalty * (mb_actions ** 2).mean(axis=2)
            print('EMP Finished.')

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

        mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, true_reward = \
            map(swap_and_flatten, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, true_reward))

        return mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, mb_states, ep_infos, true_reward

    def update_emp_history(self, emp_data):
        train_epoch = 5
        if emp_data.get('neg_log_prob') is not None:
            obs = emp_data['obs']
            actions = emp_data['actions']
            obs_t = emp_data['obs_t']
            neg_log_prob = emp_data['neg_log_prob']
            importance = -1 / neg_log_prob
            importance = importance / np.sum(importance)
            if self.emp_history is None:
                size = [len(importance)]
            else:
                size = [len(importance) // 2]
            indices = np.random.choice(np.arange(len(importance)), size=size, p=importance)
            emp_data = {'obs': obs[indices],
                        'actions': actions[indices],
                        'obs_t': obs_t[indices]}
        if self.emp_history is None:
            self.emp_history = emp_data
            train_epoch *= 5
        else:
            eval_feed_dict = {self.gce_model.obs_raw: emp_data['obs'],
                              self.gce_model.action_raw: emp_data['actions'],
                              self.gce_model.obs_raw_t: emp_data['obs_t']}
            eval_loss = self.gce_model.sess.run(self.gce_model.individual_loss, eval_feed_dict)
            select_idx = np.where(eval_loss > self.gce_train_loss)
            if len(select_idx) < 1:
                return
            obs_select = emp_data['obs'][select_idx]
            actions_select = emp_data['actions'][select_idx]
            obs_t_select = emp_data['obs_t'][select_idx]
            self.emp_history['obs'] = np.concatenate([obs_select, self.emp_history['obs']])
            self.emp_history['actions'] = np.concatenate([actions_select, self.emp_history['actions']])
            self.emp_history['obs_t'] = np.concatenate([obs_t_select, self.emp_history['obs_t']])
            if len(self.emp_history['obs']) > self.emp_history_size:
                idx = np.random.choice(len(self.emp_history['obs']), self.emp_history_size, replace=False)
                self.emp_history['obs'] = self.emp_history['obs'][idx]
                self.emp_history['actions'] = self.emp_history['actions'][idx]
                self.emp_history['obs_t'] = self.emp_history['obs_t'][idx]

        self.gce_train_loss = self.gce_model.train(self.emp_history,
                                                   batch_size=1024,
                                                   num_epoch=train_epoch,
                                                   verbose=True)

    def collect_uniform_emp_data(self, obs):
        train_epoch = 10
        if self.is_mujoco:
            if self.model.no_reset_at_all:
                mujoco_sim = []
                for e in self.env.envs:
                    qpos = e.env.unwrapped.sim.data.qpos.copy()
                    qvel = e.env.unwrapped.sim.data.qvel.copy()
                    mujoco_sim.append(np.array([qpos, qvel]))
                mujoco_sim = np.array(mujoco_sim)
                obs, action_chains, obs_t, final_sim = mp_mj_collect_input_from_state_return_final(
                    self.emp_trajectory_options, mujoco_sim)
                for i in range(len(mujoco_sim)):
                    self.env.envs[i].unwrapped.set_state(final_sim[i][0], final_sim[i][1])
            else:
                obs = np.array(obs)
                obs, action_chains, obs_t = mp_mj_collect_input_from_state(self.emp_trajectory_options, obs)
        else:
            if self.model.no_reset_at_all:
                obs, action_chains, obs_t = mp_collect_input_from_state(self.emp_trajectory_options, self.obs.copy())
                num_envs = len(self.obs)
                sample_per_env = len(obs_t) // num_envs
                ending_obs = obs_t[sample_per_env - 1::sample_per_env]
                for i in range(num_envs):
                    self.env.envs[i].unwrapped.set_state(ending_obs[i])
                self.total_random_steps += len(self.obs) * self.emp_trajectory_options['total_steps']
            else:
                obs = obs.reshape(([obs.shape[0] * obs.shape[1]] + list(obs.shape[2:])))
                idx = np.random.choice(len(obs), len(obs) // self.emp_trajectory_options['total_steps'], replace=False)
                # if self.emp_history is None:
                #     random_obs = obs
                # else:
                random_obs = obs[idx]
                obs, action_chains, obs_t = mp_collect_input_from_state(self.emp_trajectory_options, random_obs)
                self.total_random_steps += len(random_obs) * self.emp_trajectory_options['total_steps']
            print(self.total_random_steps)

        if self.emp_history is None:
            self.emp_history = {'obs': obs,
                                'actions': action_chains,
                                'obs_t': obs_t}
            train_epoch *= 5
        else:
            eval_feed_dict = {self.gce_model.obs_raw: obs,
                              self.gce_model.action_raw: action_chains,
                              self.gce_model.obs_raw_t: obs_t}
            if self.gce_model.individual_loss is not None:
                eval_loss = self.gce_model.sess.run(self.gce_model.individual_loss, eval_feed_dict)
                select_idx = np.where(eval_loss > self.gce_train_loss)
            else:
                select_idx = np.arange(len(obs))
            if len(select_idx) < 1:
                return
            obs = obs[select_idx]
            action_chains = action_chains[select_idx]
            obs_t = obs_t[select_idx]
            self.emp_history['obs'] = np.concatenate([obs, self.emp_history['obs']])
            self.emp_history['actions'] = np.concatenate([action_chains, self.emp_history['actions']])
            self.emp_history['obs_t'] = np.concatenate([obs_t, self.emp_history['obs_t']])
            if len(self.emp_history['obs']) > self.emp_history_size:
                idx = np.random.choice(len(self.emp_history['obs']), self.emp_history_size, replace=False)
                self.emp_history['obs'] = self.emp_history['obs'][idx]
                self.emp_history['actions'] = self.emp_history['actions'][idx]
                self.emp_history['obs_t'] = self.emp_history['obs_t'][idx]
        self.gce_train_loss = self.gce_model.train(self.emp_history,
                                                   batch_size=1024,
                                                   num_epoch=train_epoch,
                                                   verbose=True)


# Stable-baseline helper functions
def get_schedule_fn(value_schedule):
    if isinstance(value_schedule, (float, int)):
        value_schedule = const_fn(float(value_schedule))
    else:
        assert callable(value_schedule)
    return value_schedule


def swap_and_flatten(arr):
    shape = arr.shape
    return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])


def const_fn(val):
    def func(_):
        return val
    return func


def safe_mean(arr):
    return np.nan if len(arr) == 0 else np.mean(arr)
