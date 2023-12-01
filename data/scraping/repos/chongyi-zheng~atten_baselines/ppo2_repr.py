import time
import os
import gym
import numpy as np
import tensorflow as tf
import cv2

from stable_baselines import logger
from stable_baselines.common import explained_variance, ActorCriticRLModel, tf_util, SetVerbosity, TensorboardWriter
from stable_baselines.common.runners import AbstractEnvRunner
from stable_baselines.common.policies import ActorCriticPolicy, RecurrentActorCriticPolicy
from stable_baselines.common.schedules import get_schedule_fn
from stable_baselines.common.tf_util import total_episode_reward_logger
from stable_baselines.common.math_util import safe_mean, safe_sum
from stable_baselines.ppo2.dqn_utils import get_true_return
from attn_toy.memory.episodic_memory import EpisodicMemory
import pyvirtualdisplay


class PPO2Repr(ActorCriticRLModel):
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

    def __init__(self, policy, env, test_env=None, gamma=0.99, n_steps=128, ent_coef=0.01, learning_rate=2.5e-4,
                 vf_coef=0.5,
                 repr_coef=1., contra_coef=1., atten_encoder_coef=5 * 1. / 256, atten_decoder_coef=1.,
                 regularize_coef=1e-4,
                 max_grad_norm=0.5, lam=0.95, nminibatches=4, noptepochs=4, cliprange=0.2, cliprange_vf=None,
                 verbose=0, tensorboard_log=None, _init_setup_model=True, policy_kwargs=None,
                 full_tensorboard_log=False, seed=None, n_cpu_tf_sess=None, c_loss_type="origin", use_attention=True,
                 replay_buffer=None):

        self.learning_rate = learning_rate
        self.cliprange = cliprange
        self.cliprange_vf = cliprange_vf
        self.n_steps = n_steps
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.repr_coef = repr_coef
        self.contra_coef = contra_coef
        self.atten_encoder_coef = atten_encoder_coef
        self.atten_decoder_coef = atten_decoder_coef
        self.regularize_coef = regularize_coef
        self.max_grad_norm = max_grad_norm
        self.gamma = gamma
        self.lam = lam
        self.nminibatches = nminibatches
        self.noptepochs = noptepochs
        self.tensorboard_log = tensorboard_log
        self.full_tensorboard_log = full_tensorboard_log
        self.use_attetion = use_attention
        self.action_ph = None
        self.advs_ph = None
        self.rewards_ph = None
        self.old_neglog_pac_ph = None
        self.old_vpred_ph = None
        self.learning_rate_ph = None
        self.clip_range_ph = None
        self.mem_return_ph = None
        self.entropy = None
        self.vf_loss = None
        self.pg_loss = None
        self.pg_loss_stop_gd = None
        self.approxkl_stop_gd = None
        self.clipfrac_stop_gd = None
        self.repr_loss = None
        self.contrastive_loss = None
        self.approxkl = None
        self.approxkl_stop_gd = None
        self.clipfrac = None
        self.clipfrac_stop_gd = None
        self._train = None
        self._finetune = None
        self.loss_names = None
        self.finetune_loss_names = None
        self.train_model = None
        self.act_model = None
        self.value = None
        self.n_batch = None
        self.summary = None
        self._test_runner = None
        self.c_loss_type = c_loss_type

        self.target_model = None
        self.positive_model = None
        self.negative_model = None
        self.attention_model = None
        self.repr_coef_ph = None
        self.num_actions = env.action_space.n
        # self.replay_buffer = EpisodicMemory(10000, env.observation_space.low.shape, env.action_space.n)
        if replay_buffer is None:
            self.replay_buffer = EpisodicMemory(capacity=20000, obs_shape=env.observation_space.shape,
                                           num_actions=env.action_space.n)
        else:
            self.replay_buffer = replay_buffer

        self.magic_num = None
        # self.value_dict = value_dict
        super().__init__(policy=policy, env=env, verbose=verbose, requires_vec_env=True,
                         _init_setup_model=_init_setup_model, policy_kwargs=policy_kwargs,
                         seed=seed, n_cpu_tf_sess=n_cpu_tf_sess)
        self.policy_kwargs["add_attention"] = self.use_attetion
        self.test_env = test_env
        if _init_setup_model:
            self.setup_model()

    def _make_runner(self):
        return Runner(env=self.env, model=self, n_steps=self.n_steps,
                      gamma=self.gamma, lam=self.lam)

    def _make_test_runner(self):
        return Runner(env=self.test_env, model=self, n_steps=self.n_steps,
                      gamma=self.gamma, lam=self.lam)

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
                    assert self.n_envs % self.nminibatches == 0, "For recurrent policies, " \
                                                                 "the number of environments run in parallel should be a multiple of nminibatches."
                    n_batch_step = self.n_envs
                    n_batch_train = self.n_batch // self.nminibatches

                act_model = self.policy(self.sess, self.observation_space, self.action_space, self.n_envs, 1,
                                        n_batch_step, reuse=False, **self.policy_kwargs)

                # magic_num = tf.get_variable("magic")
                self.magic_num = tf.get_variable(name='magic', shape=[1])
                self.sess.run(self.magic_num.assign([0]))

                with tf.variable_scope("train_model", reuse=True,
                                       custom_getter=tf_util.outer_scope_getter("train_model")):
                    train_model = self.policy(self.sess, self.observation_space, self.action_space,
                                              self.n_envs // self.nminibatches, self.n_steps, n_batch_train,
                                              reuse=True, **self.policy_kwargs)
                    positive_model = self.policy(self.sess, self.observation_space, self.action_space,
                                                 self.n_envs // self.nminibatches, self.n_steps, n_batch_train,
                                                 reuse=True, **self.policy_kwargs)
                    negative_model = self.policy(self.sess, self.observation_space, self.action_space,
                                                 self.n_envs // self.nminibatches, self.n_steps, n_batch_train,
                                                 reuse=True, **self.policy_kwargs)
                    target_model = self.policy(self.sess, self.observation_space, self.action_space,
                                               self.n_envs // self.nminibatches, self.n_steps, n_batch_train,
                                               reuse=True, **self.policy_kwargs)
                    attention_model = self.policy(self.sess, self.observation_space, self.action_space,
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
                    self.mem_return_ph = tf.placeholder(tf.float32, [None, self.num_actions], name="mem_return_ph")
                    self.repr_coef_ph = tf.placeholder(tf.float32, [], name="repr_coef_ph")

                    neglogpac = train_model.proba_distribution.neglogp(self.action_ph)
                    neglogpac_stop_gd = train_model.stop_gd_proba_distribution.neglogp(self.action_ph)
                    self.entropy = tf.reduce_mean(train_model.proba_distribution.entropy())
                    self.entropy_stop_gd = tf.reduce_mean(train_model.stop_gd_proba_distribution.entropy())

                    vpred = train_model.value_flat
                    vpred_stop_gd = train_model._stop_gd_value_flat

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

                        vpred_clipped_stop_gd = self.old_vpred_ph + \
                                                tf.clip_by_value(train_model._stop_gd_value_flat - self.old_vpred_ph,
                                                                 - self.clip_range_vf_ph, self.clip_range_vf_ph)

                    vf_losses1 = tf.square(vpred - self.rewards_ph)
                    vf_losses2 = tf.square(vpred_clipped - self.rewards_ph)
                    self.vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))

                    vf_losses1_stop_gd = tf.square(vpred_stop_gd - self.rewards_ph)
                    vf_losses2_stop_gd = tf.square(vpred_clipped_stop_gd - self.rewards_ph)
                    self.vf_loss_stop_gd = .5 * tf.reduce_mean(tf.maximum(vf_losses1_stop_gd, vf_losses2_stop_gd))

                    ratio = tf.exp(self.old_neglog_pac_ph - neglogpac)
                    pg_losses = -self.advs_ph * ratio
                    pg_losses2 = -self.advs_ph * tf.clip_by_value(ratio, 1.0 - self.clip_range_ph, 1.0 +
                                                                  self.clip_range_ph)
                    self.pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
                    self.approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - self.old_neglog_pac_ph))
                    self.clipfrac = tf.reduce_mean(tf.cast(tf.greater(tf.abs(ratio - 1.0),
                                                                      self.clip_range_ph), tf.float32))

                    ratio_stop_gd = tf.exp(self.old_neglog_pac_ph - neglogpac_stop_gd)
                    pg_losses_stop_gd = -self.advs_ph * ratio_stop_gd
                    pg_losses2_stop_gd = -self.advs_ph * tf.clip_by_value(ratio_stop_gd, 1.0 - self.clip_range_ph, 1.0 +
                                                                          self.clip_range_ph)
                    self.pg_loss_stop_gd = tf.reduce_mean(tf.maximum(pg_losses_stop_gd, pg_losses2_stop_gd))
                    self.approxkl_stop_gd = .5 * tf.reduce_mean(tf.square(neglogpac_stop_gd - self.old_neglog_pac_ph))
                    self.clipfrac_stop_gd = tf.reduce_mean(tf.cast(tf.greater(tf.abs(ratio_stop_gd - 1.0),
                                                                              self.clip_range_ph), tf.float32))

                    emb_cur = target_model.contra_repr
                    emb_next = positive_model.contra_repr
                    emb_neq = negative_model.contra_repr

                    self.contrastive_loss = self.contra_coef * \
                                            (self.contrastive_loss_fc(emb_cur, emb_next, emb_neq,
                                                                      c_type=self.c_loss_type) +
                                             self.contrastive_loss_fc(emb_next, emb_cur, emb_neq,
                                                                      c_type=self.c_loss_type))

                    attention_max = tf.reduce_max(attention_model.attention, axis=1, keepdims=True)
                    attention_min = tf.reduce_min(attention_model.attention, axis=1, keepdims=True)
                    normalized_attention = (attention_model.attention - attention_min) / (
                            attention_max - attention_min + 1e-12)
                    self.encoder_loss = self.atten_encoder_coef * tf.reduce_mean(
                        tf.norm(normalized_attention, ord=1, axis=1))
                    # self.weight_loss = tf.norm(self.train_model.weighted_w,ord=1)
                    self.decoder_loss = self.atten_decoder_coef * tf.reduce_mean(
                        tf.square(attention_model.mem_value_fn - self.mem_return_ph))

                    self.repr_loss = self.contrastive_loss + self.encoder_loss + self.decoder_loss
                    loss = self.pg_loss - self.entropy * self.ent_coef + self.vf_loss * self.vf_coef + self.repr_coef * self.repr_loss
                    ppo_loss = self.pg_loss - self.entropy * self.ent_coef + self.vf_loss * self.vf_coef

                    loss_stop_gd = self.pg_loss_stop_gd - self.entropy_stop_gd * self.ent_coef + self.vf_loss_stop_gd * self.vf_coef

                    tf.summary.scalar('entropy_loss', self.entropy)
                    tf.summary.scalar('repr_loss', self.repr_loss)
                    tf.summary.scalar('policy_gradient_loss', self.pg_loss)
                    tf.summary.scalar('policy_gradient_loss_finetune', self.pg_loss_stop_gd)
                    tf.summary.scalar('value_function_loss', self.vf_loss)
                    tf.summary.scalar('value_function_loss_finetune', self.vf_loss_stop_gd)
                    tf.summary.scalar('approximate_kullback-leibler', self.approxkl)
                    tf.summary.scalar('approximate_kullback-leibler_finetune', self.approxkl_stop_gd)
                    tf.summary.scalar('clip_factor', self.clipfrac)
                    tf.summary.scalar('clip_factor_finetune', self.clipfrac_stop_gd)
                    tf.summary.scalar('loss', loss)
                    tf.summary.scalar('loss_finetune', loss_stop_gd)
                    tf.summary.scalar('contrastive_loss', self.contrastive_loss)
                    tf.summary.scalar('atten_encoder_loss', self.encoder_loss)
                    tf.summary.scalar('atten_decoder_loss', self.decoder_loss)

                    with tf.variable_scope('model'):
                        self.params = tf.trainable_variables()
                        print(self.params)
                        self.ppo_params = tf.trainable_variables(scope="model/last_layer")
                        self.repr_params = tf.trainable_variables(scope="model/feature")
                        print(self.ppo_params, self.repr_params)
                        if self.full_tensorboard_log:
                            for var in self.params:
                                tf.summary.histogram(var.name, var)

                    l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.repr_params]) * self.regularize_coef
                    # l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.repr_params
                    #                     if 'bias' not in v.name]) * self.regularize_coef
                    grads = tf.gradients(ppo_loss, self.params)
                    if self.max_grad_norm is not None:
                        grads, _grad_norm = tf.clip_by_global_norm(grads, self.max_grad_norm)
                    grads = list(zip(grads, self.params))

                    repr_grads = tf.gradients(self.repr_coef_ph * (self.repr_loss + l2_loss), self.repr_params)
                    # if self.max_grad_norm is not None:
                    #     repr_grads, _repr_grad_norm = tf.clip_by_global_norm(repr_grads, self.max_grad_norm)
                    repr_grads = list(zip(repr_grads, self.repr_params))

                    grads_stop_gd = tf.gradients(loss_stop_gd, self.params)
                    if self.max_grad_norm is not None:
                        grads_stop_gd, _grad_norm_stop_gd = tf.clip_by_global_norm(grads_stop_gd, self.max_grad_norm)
                    grads_stop_gd = list(zip(grads_stop_gd, self.params))

                trainer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph, epsilon=1e-5)
                repr_trainer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph, epsilon=1e-5)
                finetuner = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph, epsilon=1e-5)
                self._train = trainer.apply_gradients(grads)
                self._repr_train = repr_trainer.apply_gradients(repr_grads)
                self._finetune = finetuner.apply_gradients(grads_stop_gd)

                self.loss_names = ['repr_loss', 'contrastive_loss', 'atten_encoder_loss', 'atten_decoder_loss',
                                   'policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac']
                self.finetune_loss_names = ['repr_loss_finetune', 'policy_loss_finetune', 'value_loss_finetune',
                                            'policy_entropy_finetune', 'approxkl_finetune', 'clipfrac_finetune']

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
                self.target_model = target_model
                self.positive_model = positive_model
                self.negative_model = negative_model
                self.attention_model = attention_model
                self.act_model = act_model
                self.step = act_model.step
                # self.act = act_model.act
                self.proba_step = act_model.proba_step
                self.value = act_model.value
                self.initial_state = act_model.initial_state
                tf.global_variables_initializer().run(session=self.sess)  # pylint: disable=E1101

                self.summary = tf.summary.merge_all()

    def _train_step(self, learning_rate, cliprange, obs_target, obs_pos, obs_neg, obs_value, true_returns_target, obs,
                    returns,
                    masks, actions, values,
                    neglogpacs, update,
                    writer, states=None, cliprange_vf=None,repr_coef=1.):
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
                  self.old_neglog_pac_ph: neglogpacs, self.old_vpred_ph: values,
                  self.target_model.obs_ph: obs_target,
                  self.attention_model.obs_ph: obs_value,
                  self.positive_model.obs_ph: obs_pos,
                  self.negative_model.obs_ph: obs_neg,
                  self.mem_return_ph: true_returns_target,
                  self.repr_coef_ph : repr_coef}
        # print(true_returns_target)
        if states is not None:
            td_map[self.train_model.states_ph] = states
            td_map[self.train_model.dones_ph] = masks

        if cliprange_vf is not None and cliprange_vf >= 0:
            td_map[self.clip_range_vf_ph] = cliprange_vf

        if states is None:
            update_fac = max(self.n_batch // self.nminibatches // self.noptepochs, 1)
        else:
            update_fac = max(self.n_batch // self.nminibatches // self.noptepochs // self.n_steps, 1)

        if writer is not None:
            # run loss backprop with summary, but once every 10 runs save the metadata (memory, compute time, ...)
            if self.full_tensorboard_log and (1 + update) % 10 == 0:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, repr_loss, contrastive_loss, atten_encoder_loss, atten_decoder_loss, policy_loss, value_loss, policy_entropy, approxkl, clipfrac, _, _ = self.sess.run(
                    [self.summary, self.repr_loss, self.contrastive_loss, self.encoder_loss, self.decoder_loss,
                     self.pg_loss, self.vf_loss, self.entropy, self.approxkl,
                     self.clipfrac, self._repr_train, self._train],
                    td_map, options=run_options, run_metadata=run_metadata)
                writer.add_run_metadata(run_metadata, 'step%d' % (update * update_fac))
            else:
                summary, repr_loss, contrastive_loss, atten_encoder_loss, atten_decoder_loss, policy_loss, value_loss, policy_entropy, approxkl, clipfrac, _, _ = self.sess.run(
                    [self.summary, self.repr_loss, self.contrastive_loss, self.encoder_loss, self.decoder_loss,
                     self.pg_loss, self.vf_loss, self.entropy, self.approxkl,
                     self.clipfrac, self._repr_train, self._train],
                    td_map)
            writer.add_summary(summary, (update * update_fac))
        else:
            repr_loss, contrastive_loss, atten_encoder_loss, atten_decoder_loss, policy_loss, value_loss, policy_entropy, approxkl, clipfrac, _, _ = self.sess.run(
                [self.repr_loss, self.contrastive_loss, self.encoder_loss, self.decoder_loss, self.pg_loss,
                 self.vf_loss, self.entropy, self.approxkl, self.clipfrac, self._repr_train, self._train],
                td_map)

        return repr_loss, contrastive_loss, atten_encoder_loss, atten_decoder_loss, policy_loss, value_loss, policy_entropy, approxkl, clipfrac

    def _finetune_step(self, learning_rate, cliprange, obs, returns, masks, actions, values, neglogpacs, update,
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
            update_fac = max(self.n_batch // self.nminibatches // self.noptepochs, 1)
        else:
            update_fac = max(self.n_batch // self.nminibatches // self.noptepochs // self.n_steps, 1)

        if writer is not None:
            # run loss backprop with summary, but once every 10 runs save the metadata (memory, compute time, ...)
            if self.full_tensorboard_log and (1 + update) % 10 == 0:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, policy_loss, value_loss, policy_entropy, approxkl, clipfrac, _ = self.sess.run(
                    [self.summary, self.pg_loss_stop_gd, self.vf_loss_stop_gd, self.entropy_stop_gd,
                     self.approxkl_stop_gd,
                     self.clipfrac, self._train],
                    td_map, options=run_options, run_metadata=run_metadata)
                writer.add_run_metadata(run_metadata, 'step%d' % (update * update_fac))
            else:
                summary, policy_loss, value_loss, policy_entropy, approxkl, clipfrac, _ = self.sess.run(
                    [self.summary, self.pg_loss_stop_gd, self.vf_loss_stop_gd, self.entropy_stop_gd,
                     self.approxkl_stop_gd,
                     self.clipfrac_stop_gd, self._finetune],
                    td_map)
            writer.add_summary(summary, (update * update_fac))
        else:
            policy_loss, value_loss, policy_entropy, approxkl, clipfrac, _ = self.sess.run(
                [self.pg_loss_stop_gd, self.vf_loss_stop_gd, self.entropy_stop_gd, self.approxkl_stop_gd,
                 self.clipfrac_stop_gd, self._finetune],
                td_map)

        return 0, policy_loss, value_loss, policy_entropy, approxkl, clipfrac

    @property
    def test_runner(self) -> AbstractEnvRunner:
        if self._test_runner is None:
            self._test_runner = self._make_test_runner()
        return self._test_runner

    def eval(self, tb_log_name="PPO2", callback=None, print_attention_map=False, filedir=None):
        new_tb_log = self._init_num_timesteps(False)
        callback = self._init_callback(callback)
        runner = self.test_runner  # run on eval env
        with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name, new_tb_log) \
                as writer:
            self._setup_test()
            t_start = time.time()
            callback.on_training_start(locals(), globals())

            rollout = runner.run(callback)
            # Unpack
            obs, returns, masks, actions, values, neglogpacs, states, ep_infos, true_reward, attention, feature_map = rollout
            self.ep_info_buf_test.extend(ep_infos)
            t_now = time.time()
            fps = int(self.n_batch / (t_now - t_start))

            if writer is not None:
                total_episode_reward_logger(self.episode_reward_test,
                                            true_reward.reshape((self.n_envs, self.n_steps)),
                                            masks.reshape((self.n_envs, self.n_steps)),
                                            writer, self.num_timesteps, suffix="_eval")

            if self.verbose >= 1:
                logger.logkv("total_timesteps_eval", self.num_timesteps)
                logger.logkv("fps_eval", fps)
                if len(self.ep_info_buf_test) > 0 and len(self.ep_info_buf_test[0]) > 0:
                    logger.logkv('ep_reward_mean_eval', safe_mean([ep_info['r'] for ep_info in self.ep_info_buf_test]))
                    logger.logkv('ep_len_mean_eval', safe_mean([ep_info['l'] for ep_info in self.ep_info_buf_test]))
                logger.logkv('time_elapsed_eval', t_now - t_start)
                logger.dumpkvs()

            # save attention image
            if print_attention_map:
                if filedir is None:
                    filedir = os.getenv('OPENAI_LOGDIR')
                    filedir = os.path.join(filedir, "attention_eval")
                rnd_indices = np.random.choice(len(obs), 5)
                for i in range(len(rnd_indices)):
                    ind = rnd_indices[i]

                    self.save_attention(attention[ind], obs[ind], feature_map[i], filedir, self.num_timesteps, i)

        callback.on_training_end()
        return self

    def restore_map(self, flatten_map, obs_shape):
        length = int(np.sqrt(np.size(flatten_map)))
        flatten_map = flatten_map.reshape(length, length)
        flatten_map = (flatten_map - np.min(flatten_map)) / (np.max(flatten_map) - np.min(flatten_map))
        flatten_map = cv2.resize(flatten_map, (obs_shape[0], obs_shape[1]))

        flatten_map = np.repeat(flatten_map[..., np.newaxis], 3, axis=2)
        return flatten_map

    def save_attention(self, attention, obs, feature_map, subdir, step, num):
        # subdir = os.path.join(filedir, "./attention")
        # print(attention.squeeze())

        attention = self.restore_map(attention, obs.shape)
        feature_map = self.restore_map(feature_map, obs.shape)
        image = np.array(obs)[..., :3]
        # print(image.shape)
        attentioned_image = image * attention
        if not os.path.isdir(subdir):
            os.makedirs(os.path.join(subdir, "./mask/"))
            os.makedirs(os.path.join(subdir, "./masked_image/"))
            os.makedirs(os.path.join(subdir, "./image/"))
            os.makedirs(os.path.join(subdir, "./feature_map/"))
        # print(attention.shape)
        cv2.imwrite(os.path.join(subdir, "./masked_image/", "masked_image_{}_{}.png".format(step, num)),
                    attentioned_image.transpose((1, 0, 2)))
        # attentioned_image)
        cv2.imwrite(os.path.join(subdir, "./mask/", "attention_{}_{}.png".format(step, num)),
                    # attention * 255)
                    attention.transpose((1, 0, 2)) * 255)

        cv2.imwrite(os.path.join(subdir, "./feature_map/", "feature_map_{}_{}.png".format(step, num)),
                    # attention * 255)
                    feature_map.transpose((1, 0, 2)) * 255)

        cv2.imwrite(os.path.join(subdir, "./image/", "obs_{}_{}.png".format(step, num)),
                    # image * 255)
                    image.transpose((1, 0, 2)))

    def learn(self, total_timesteps, repr_coef=1.,finetune=False, callback=None, log_interval=1, tb_log_name="PPO2",
              reset_num_timesteps=True, begin_eval=False, print_attention_map=False, filedir=None):
        # Transform to callable if needed
        self.learning_rate = get_schedule_fn(self.learning_rate)
        self.cliprange = get_schedule_fn(self.cliprange)
        cliprange_vf = get_schedule_fn(self.cliprange_vf)

        new_tb_log = self._init_num_timesteps(reset_num_timesteps)
        callback = self._init_callback(callback)
        if begin_eval:
            self.replay_buffer.empty()

        runner = self.runner if not finetune else self.test_runner
        with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name, new_tb_log) \
                as writer:
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
                rollout = runner.run(callback)
                # Unpack
                obs, returns, masks, actions, values, neglogpacs, states, ep_infos, true_reward, attention, feature_map = rollout
                true_returns = get_true_return(true_reward, masks, self.n_envs)
                # if self.value_dict is not None:
                #     true_returns = np.array(
                #         [self.value_dict.get(obs[i + 1].data.tobytes(), np.nan) for i in range(len(obs) - 1)] + [
                #             np.nan])

                # find_value_rate = (np.sum(np.isnan(true_returns)) + 0.0) / len(true_returns)
                # Save
                self.replay_buffer.add_batch(obs, actions, true_reward, true_returns, masks)

                print("not nan percentage:", self.replay_buffer.percentage(), self.replay_buffer.curr_capacity)
                callback.on_rollout_end()

                # Early stopping due to the callback
                if not runner.continue_training:
                    break

                self.ep_info_buf.extend(ep_infos)
                mb_loss_vals = []
                if states is None:  # nonrecurrent version
                    update_fac = max(self.n_batch // self.nminibatches // self.noptepochs, 1)
                    inds = np.arange(self.n_batch)
                    for epoch_num in range(self.noptepochs):
                        np.random.shuffle(inds)
                        for start in range(0, self.n_batch, batch_size):
                            timestep = self.num_timesteps // update_fac + ((epoch_num *
                                                                            self.n_batch + start) // batch_size)
                            end = start + batch_size
                            mbinds = inds[start:end]
                            if not finetune:
                                obs_t_batch, act_t_batch, rew_t_batch, obs_tp1_batch, done_mask_batch, obs_neg_batch, _ = self.replay_buffer.sample(
                                    len(mbinds))
                                obs_value_batch, _, _, _, _, _, return_t_batch = self.replay_buffer.sample(
                                    len(mbinds))
                                predict_returns = self.sess.run(self.attention_model.mem_value_fn,
                                                                {self.attention_model.obs_ph: obs_value_batch})
                                predict_returns = predict_returns.reshape(-1)

                                return_t_batch = np.array(return_t_batch)
                                shape = return_t_batch.shape
                                return_t_batch = return_t_batch.reshape(-1)
                                return_t_batch[np.isinf(return_t_batch)] = predict_returns[np.isinf(return_t_batch)]
                                return_t_batch = return_t_batch.reshape(shape)

                            slices = (arr[mbinds] for arr in
                                      (obs, returns, masks, actions, values, neglogpacs))

                            if finetune:
                                mb_loss_vals.append(
                                    self._finetune_step(lr_now, cliprange_now,
                                                        *slices, writer=writer,
                                                        update=timestep, cliprange_vf=cliprange_vf_now))
                            else:
                                mb_loss_vals.append(
                                    self._train_step(lr_now, cliprange_now, obs_t_batch, obs_tp1_batch, obs_neg_batch,
                                                     obs_value_batch, return_t_batch,
                                                     *slices, writer=writer,
                                                     update=timestep, cliprange_vf=cliprange_vf_now,repr_coef=repr_coef))
                else:  # recurrent version
                    update_fac = max(self.n_batch // self.nminibatches // self.noptepochs // self.n_steps, 1)
                    assert self.n_envs % self.nminibatches == 0
                    env_indices = np.arange(self.n_envs)
                    flat_indices = np.arange(self.n_envs * self.n_steps).reshape(self.n_envs, self.n_steps)
                    envs_per_batch = batch_size // self.n_steps
                    for epoch_num in range(self.noptepochs):
                        np.random.shuffle(env_indices)
                        for start in range(0, self.n_envs, envs_per_batch):
                            timestep = self.num_timesteps // update_fac + ((epoch_num *
                                                                            self.n_envs + start) // envs_per_batch)
                            end = start + envs_per_batch
                            mb_env_inds = env_indices[start:end]
                            mb_flat_inds = flat_indices[mb_env_inds].ravel()
                            if not finetune:
                                obs_t_batch, act_t_batch, rew_t_batch, obs_tp1_batch, done_mask_batch, obs_neg_batch, _ = self.replay_buffer.sample(
                                    len(mbinds))
                                obs_value_batch, _, _, _, _, _, return_t_batch = self.replay_buffer.sample(
                                    len(mbinds))
                                predict_returns = self.sess.run(self.attention_model.mem_value_fn,
                                                                {self.attention_model.obs_ph: obs_value_batch})
                                predict_returns = predict_returns.reshape(-1)
                                return_t_batch[np.isinf(return_t_batch)] = predict_returns[np.isinf(return_t_batch)]

                            slices = (arr[mb_flat_inds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                            mb_states = states[mb_env_inds]
                            if finetune:

                                mb_loss_vals.append(
                                    self._finetune_step(lr_now, cliprange_now,
                                                        *slices, update=timestep,
                                                        writer=writer, states=mb_states,
                                                        cliprange_vf=cliprange_vf_now))
                            else:
                                mb_loss_vals.append(
                                    self._train_step(lr_now, cliprange_now, obs_t_batch, obs_tp1_batch, obs_neg_batch,
                                                     obs_value_batch, return_t_batch, *slices, update=timestep,
                                                     writer=writer, states=mb_states,
                                                     cliprange_vf=cliprange_vf_now,repr_coef=repr_coef))

                loss_vals = np.mean(mb_loss_vals, axis=0)
                t_now = time.time()
                fps = int(self.n_batch / (t_now - t_start))
                suffix = "_finetune" if finetune else ""

                # print("test tensorboard",self.tensorboard_log,writer)

                if writer is not None:
                    total_episode_reward_logger(self.episode_reward,
                                                true_reward.reshape((self.n_envs, self.n_steps)),
                                                masks.reshape((self.n_envs, self.n_steps)),
                                                writer, self.num_timesteps)

                if self.verbose >= 1 and (update % log_interval == 0 or update == 1):
                    explained_var = explained_variance(values, returns)
                    logger.logkv("serial_timesteps" + suffix, update * self.n_steps)
                    logger.logkv("n_updates" + suffix, update)
                    logger.logkv("total_timesteps" + suffix, self.num_timesteps)
                    # logger.logkv("find_value_rate" + suffix, find_value_rate)
                    logger.logkv("fps" + suffix, fps)
                    logger.logkv("explained_variance" + suffix, float(explained_var))
                    if len(self.ep_info_buf) > 0 and len(self.ep_info_buf[0]) > 0:
                        logger.logkv('ep_reward_mean' + suffix,
                                     safe_mean([ep_info['r'] for ep_info in self.ep_info_buf]))
                        logger.logkv('ep_len_mean' + suffix, safe_mean([ep_info['l'] for ep_info in self.ep_info_buf]))
                    logger.logkv('time_elapsed' + suffix, t_start - t_first_start)
                    loss_names = self.loss_names if not finetune else self.finetune_loss_names
                    for (loss_val, loss_name) in zip(loss_vals, loss_names):
                        logger.logkv(loss_name, loss_val)
                    logger.dumpkvs()

                if print_attention_map:
                    rnd_indices = np.random.choice(len(obs), 5)
                    if filedir is None:
                        filedir = os.getenv('OPENAI_LOGDIR')
                        filedir = os.path.join(filedir, "attention_train")
                    for i in range(len(rnd_indices)):
                        ind = rnd_indices[i]
                        self.save_attention(attention[ind], obs[ind], feature_map[i], filedir, self.num_timesteps, i)

            callback.on_training_end()
            return self

    def save(self, save_path, cloudpickle=False):

        self.sess.run(self.magic_num.assign([142857]))
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

    def act(self, obs, is_train=False):
        action, value, self.initial_state, neglogp, attention_saved, feature_map = self.step(obs, None, None,
                                                                                             deterministic=is_train)
        self.num_timesteps += 1
        # print(neglogp)
        # print(action)
        print("action probability", self.act_model.proba_step(obs))
        print("random action", action)
        self.save_attention(attention_saved, obs[0], feature_map, os.getenv('OPENAI_LOGDIR'), self.num_timesteps, 0)
        return action

    # to be compatible
    def observe(self, action, reward, obs, done, train=True):
        return

    def test_magic_num(self):
        magic = self.sess.run(self.magic_num)
        print("magic number:", magic)


class Runner(AbstractEnvRunner):
    def __init__(self, *, env, model, n_steps, gamma, lam):
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
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs, mb_attention = [], [], [], [], [], [], []
        mb_featuremap = []
        mb_states = self.states
        ep_infos = []
        for _ in range(self.n_steps):
            actions, values, self.states, neglogpacs, attention, feature_map = self.model.step(self.obs, self.states,
                                                                                               self.dones)
            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)
            mb_attention.append(attention)
            mb_featuremap.append(feature_map)
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.env.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.env.action_space.low, self.env.action_space.high)
            self.obs[:], rewards, self.dones, infos = self.env.step(clipped_actions)

            self.model.num_timesteps += self.n_envs

            if self.callback is not None:
                # Abort training early
                self.callback.update_locals(locals())
                if self.callback.on_step() is False:
                    self.continue_training = False
                    # Return dummy values
                    return [None] * 9

            for info in infos:
                maybe_ep_info = info.get('episode')
                if maybe_ep_info is not None:
                    ep_infos.append(maybe_ep_info)
            mb_rewards.append(rewards)
        # batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        mb_attention = np.asarray(mb_attention, dtype=np.float32)
        mb_featuremap = np.asarray(mb_featuremap, dtype=np.float32)
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
        # print(mb_rewards)
        mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, true_reward, mb_attention, mb_featuremap = \
            map(swap_and_flatten,
                (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, true_reward, mb_attention,
                 mb_featuremap))

        return mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, mb_states, ep_infos, true_reward, mb_attention, mb_featuremap


# obs, returns, masks, actions, values, neglogpacs, states = runner.run()
def swap_and_flatten(arr):
    """
    swap and then flatten axes 0 and 1

    :param arr: (np.ndarray)
    :return: (np.ndarray)
    """
    shape = arr.shape
    return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])
