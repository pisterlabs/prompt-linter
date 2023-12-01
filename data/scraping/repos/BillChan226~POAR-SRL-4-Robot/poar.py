import time
import sys
import multiprocessing
from collections import deque

import gym
import numpy as np
import tensorflow as tf

from stable_baselines import logger
from stable_baselines.common import explained_variance, ActorCriticRLModel, tf_util, SetVerbosity, TensorboardWriter
from stable_baselines.common.runners import AbstractEnvRunner
from stable_baselines.poar.policies import SRLActorCriticPolicy
from stable_baselines.a2c.utils import total_episode_reward_logger
from stable_baselines.poar.utils import pca
from ipdb import set_trace as tt
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
#from state_representation.episode_saver import LogRLStates
sns.set()

class POAR(ActorCriticRLModel):
    """
    Proximal Policy Optimization algorithm (GPU version).
    Paper: https://arxiv.org/abs/1707.06347

    :param policy: (ActorCriticPolicy or str) The policy model to use (MlpPolicy, CnnPolicy, CnnLstmPolicy, ...)
    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :parma srl_weight: (OrderDict) to save the weight of different srl_model
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
    """

    def __init__(
            self, policy, env, srl_weight, gamma=0.99, n_steps=128, ent_coef=0.01, learning_rate=2.5e-4, vf_coef=0.5,
            srl_lr=0.0001, max_grad_norm=0.5, lam=0.95, nminibatches=4, noptepochs=4, cliprange=0.2, cliprange_vf=None,
            split_dim=200, verbose=0, tensorboard_log='/home/tete/work/robotics-rl-srl/Tensorboard_logs', _init_setup_model=True, policy_kwargs=None,
            full_tensorboard_log=True):

        super(
            POAR, self).__init__(policy=policy, env=env, verbose=verbose, requires_vec_env=True,
                                 _init_setup_model=_init_setup_model, policy_kwargs=policy_kwargs)
        self.learning_rate = learning_rate
        self.cliprange = cliprange
        self.cliprange_vf = cliprange_vf
        self.n_steps = n_steps
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.gamma = gamma
        self.lam = lam
        self.split_dim = split_dim
        self.srl_weight = srl_weight
        self.nminibatches = nminibatches
        self.noptepochs = noptepochs
        self.tensorboard_log = tensorboard_log
        self.full_tensorboard_log = full_tensorboard_log
        self.srl_lr = srl_lr

        self.graph = None
        self.sess = None
        self.action_ph = None
        self.advs_ph = None
        self.rewards_ph = None
        self.old_neglog_pac_ph = None
        self.old_vpred_ph = None
        self.learning_rate_ph = None
        self.clip_range_ph = None
        self.clip_range_vf_ph = None
        self.reconstruct_ph = None
        self.true_reward_ph = None
        self.srl_lr_ph = None
        self.entropy = None
        self.vf_loss = None
        self.pg_loss = None
        self.ae_loss = None
        self.approxkl = None
        self.clipfrac = None
        self.params = None
        self._train = None
        self._srl_train = None
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
        self.srl_loss_list = None

        if _init_setup_model:
            self.setup_model()

    def _get_pretrain_placeholders(self):
        policy = self.act_model
        if isinstance(self.action_space, gym.spaces.Discrete):
            return policy.obs_ph, self.action_ph, policy.policy
        return policy.obs_ph, self.action_ph, policy.deterministic_action

    def setup_model(self):
        with SetVerbosity(self.verbose):

            assert issubclass(self.policy, SRLActorCriticPolicy), "Error: the input policy for the POAR model must be" \
                "an instance of common.policies.ActorCriticPolicy."

            self.n_batch = self.n_envs * self.n_steps

            n_cpu = multiprocessing.cpu_count()
            if sys.platform == 'darwin':
                n_cpu //= 2

            self.graph = tf.Graph()
            with self.graph.as_default():
                self.sess = tf_util.make_session(
                    num_cpu=n_cpu, graph=self.graph)
                n_batch_step = None
                n_batch_train = None
                # TODO: Not implement for RecurrentActorCritic
                # if issubclass(self.policy, RecurrentActorCriticPolicy):
                #     assert self.n_envs % self.nminibatches == 0, "For recurrent policies, "\
                #         "the number of environments run in parallel should be a multiple of nminibatches."
                #     n_batch_step = self.n_envs
                #     n_batch_train = self.n_batch // self.nminibatches

                act_model = self.policy(
                    self.sess,
                    self.observation_space,
                    self.action_space,
                    self.n_envs,
                    1,
                    n_batch_step,
                    split_dim=self.split_dim,
                    reuse=False,
                    **self.policy_kwargs)
                with tf.variable_scope("train_model", reuse=True,
                                       custom_getter=tf_util.outer_scope_getter("train_model")):
                    train_model = self.policy(
                        self.sess,
                        self.observation_space,
                        self.action_space,
                        self.n_envs //
                        self.nminibatches,
                        self.n_steps,
                        n_batch_train,
                        split_dim=self.split_dim,
                        reuse=True,
                        **self.policy_kwargs)

                with tf.variable_scope("loss", reuse=False):
                    self.action_ph = train_model.pdtype.sample_placeholder([None], name="action_ph")
                    self.advs_ph = tf.placeholder(tf.float32, [None], name="advs_ph")
                    self.rewards_ph = tf.placeholder(tf.float32, [None], name="rewards_ph")
                    self.old_neglog_pac_ph = tf.placeholder(tf.float32, [None], name="old_neglog_pac_ph")
                    self.old_vpred_ph = tf.placeholder(tf.float32, [None], name="old_vpred_ph")
                    self.learning_rate_ph = tf.placeholder(tf.float32, [], name="learning_rate_ph")
                    self.clip_range_ph = tf.placeholder(tf.float32, [], name="clip_range_ph")
                    self.true_reward_ph = tf.placeholder(tf.int32, [None], name="true_reward_ph")
                    self.srl_lr_ph = tf.placeholder(tf.float32, [], name="srl_weight_ph")
                    # self.reconstruct_ph = tf.placeholder(tf.float32, train_model.processed_obs.shape,
                    #                                      name="reconstruction_ph")
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
                        # Original PPO implementation: no value function
                        # clipping
                        self.clip_range_vf_ph = None
                    else:
                        # Last possible behavior: clipping range
                        # specific to the value function
                        self.clip_range_vf_ph = tf.placeholder(
                            tf.float32, [], name="clip_range_vf_ph")

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
                    self.vf_loss = .5 * \
                        tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
                    ratio = tf.exp(self.old_neglog_pac_ph - neglogpac)
                    pg_losses = -self.advs_ph * ratio
                    pg_losses2 = -self.advs_ph * \
                        tf.clip_by_value(ratio, 1.0 - self.clip_range_ph, 1.0 + self.clip_range_ph)
                    self.pg_loss = tf.reduce_mean(
                        tf.maximum(pg_losses, pg_losses2))
                    self.approxkl = .5 * \
                        tf.reduce_mean(tf.square(neglogpac - self.old_neglog_pac_ph))
                    self.clipfrac = tf.reduce_mean(tf.cast(tf.greater(tf.abs(ratio - 1.0), self.clip_range_ph),
                                                           tf.float32))

                    self.loss_names = [
                        'policy_loss',
                        'value_loss',
                        'policy_entropy',
                        'approxkl',
                        'clipfrac']
                    loss = (self.pg_loss - self.entropy * self.ent_coef + self.vf_loss * self.vf_coef)

                    # SRL loss
                    weight_dict = {}
                    srl_loss_dict = {}
                    # reconstruction loss
                    if "autoencoder" in self.srl_weight:
                        weight_dict["reconstruction_loss"] = self.srl_weight["autoencoder"]
                        ae_loss = tf.square(train_model.processed_obs - train_model.reconstruct_obs)
                        ae_loss += tf.square(train_model.next_processed_obs - train_model.next_reconstruct_obs)
                        ae_loss = 0.5 * tf.reduce_mean(ae_loss)
                        srl_loss_dict["reconstruction_loss"] = ae_loss
                    if "inverse" in self.srl_weight and self.srl_weight["inverse"] > 0:
                        # TODO: a continuous version should be implemented
                        weight_dict["inverse_loss"] = self.srl_weight["inverse"]
                        # inverse_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                        #     labels=tf.one_hot(
                        #         self.action_ph,
                        #         self.action_space.n),
                        #     logits=train_model.srl_action))
                        inverse_loss = tf.reduce_mean(tf.square(tf.one_hot(self.action_ph, self.action_space.n)
                                                                - train_model.srl_action))
                        srl_loss_dict["inverse_loss"] = inverse_loss
                    if "forward" in self.srl_weight and self.srl_weight["forward"] > 0:
                        weight_dict["forward_loss"] = self.srl_weight["forward"]
                        forward_loss = tf.reduce_mean(tf.square(train_model.next_latent_obs - train_model.srl_state))
                        srl_loss_dict["forward_loss"] = forward_loss
                    if "reward" in self.srl_weight and self.srl_weight["reward"] > 0:
                        weight_dict["reward_loss"] = self.srl_weight["reward"]
                        clip_reward = tf.one_hot(tf.cast(tf.clip_by_value(self.true_reward_ph, clip_value_min=0,
                                                                          clip_value_max=9), tf.int32), 10)
                        # clip_reward = tf.one_hot(tf.cast(self.true_reward_ph+1, tf.int32), 3)

                        reward_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                            logits=train_model.srl_reward, labels=clip_reward))
                        srl_loss_dict["reward_loss"] = reward_loss
                    if "entropy" in self.srl_weight and self.srl_weight["entropy"] > 0:
                        # TODO should be as a hyper parameters for Cd (coming from COAR)
                        weight_dict["state_entropy_loss"] = self.srl_weight["entropy"]
                        cd = 5.0
                        state_entropy_loss = tf.exp(-cd * tf.norm(
                            train_model.latent_obs - train_model.next_latent_obs))
                        state_entropy_loss += tf.math.maximum(
                            tf.math.maximum(tf.norm(train_model.latent_obs, ord=np.inf)**2 - 1,
                                            tf.norm(train_model.next_latent_obs, ord=np.inf)**2 - 1), 0)
                        srl_loss_dict["state_entropy_loss"] = state_entropy_loss


                    self.srl_loss_list = []
                    srl_loss = 0 #+0.001 * loss
                    for srl_loss_name in weight_dict:
                        if weight_dict[srl_loss_name] > 0:
                            srl_loss += weight_dict[srl_loss_name] * \
                                srl_loss_dict[srl_loss_name]
                            self.srl_loss_list.append(srl_loss_dict[srl_loss_name])
                            self.loss_names.append(srl_loss_name)
                            tf.summary.scalar(srl_loss_name, srl_loss_dict[srl_loss_name])
                    tf.summary.scalar('entropy_loss', self.entropy)
                    tf.summary.scalar('policy_gradient_loss', self.pg_loss)
                    tf.summary.scalar('value_function_loss', self.vf_loss)
                    tf.summary.scalar('approximate_kullback-leibler', self.approxkl)
                    tf.summary.scalar('clip_factor', self.clipfrac)
                    tf.summary.scalar('srl_loss', srl_loss)
                    tf.summary.scalar('loss', loss)

                    with tf.variable_scope('model'):
                        self.params = tf.trainable_variables()
                        if self.full_tensorboard_log:
                            for var in self.params:
                                tf.summary.histogram(var.name, var)
                    grads, srl_grads = tf.gradients(loss, self.params), tf.gradients(srl_loss, self.params)
                    # we do not optimize the encoder part, to protect the SRL structure
                    for i, g in enumerate(srl_grads):
                        if (g is not None) and (grads[i] is not None):
                            grads[i] *= 0
                    if self.max_grad_norm is not None:
                        grads, _grad_norm = tf.clip_by_global_norm(grads, self.max_grad_norm)
                    grads, srl_grads = list(zip(grads, self.params)), list(zip(srl_grads, self.params))
                srl_trainer = tf.train.AdamOptimizer(learning_rate=self.srl_lr_ph, epsilon=1e-5)
                trainer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph, epsilon=1e-5)

                self._train = trainer.apply_gradients(grads)
                self._srl_train = srl_trainer.apply_gradients(srl_grads)
                with tf.variable_scope("input_info", reuse=False):
                    tf.summary.scalar('discounted_rewards', tf.reduce_mean(self.rewards_ph))
                    tf.summary.scalar('learning_rate', tf.reduce_mean(self.learning_rate_ph))
                    tf.summary.scalar('SRL_learning_rate', tf.reduce_mean(self.srl_lr_ph))
                    tf.summary.scalar('advantage', tf.reduce_mean(self.advs_ph))
                    tf.summary.scalar('clip_range', tf.reduce_mean(self.clip_range_ph))
                    if self.clip_range_vf_ph is not None:
                        tf.summary.scalar('clip_range_vf', tf.reduce_mean(self.clip_range_vf_ph))

                    tf.summary.scalar('old_neglog_action_probabilty', tf.reduce_mean(self.old_neglog_pac_ph))
                    tf.summary.scalar('old_value_pred', tf.reduce_mean(self.old_vpred_ph))

                    if self.full_tensorboard_log:
                        tf.summary.histogram('discounted_rewards', self.rewards_ph)
                        tf.summary.histogram('learning_rate', self.learning_rate_ph)
                        tf.summary.histogram('advantage', self.advs_ph)
                        tf.summary.histogram('clip_range', self.clip_range_ph)
                        tf.summary.histogram('old_neglog_action_probabilty', self.old_neglog_pac_ph)
                        tf.summary.histogram('old_value_pred', self.old_vpred_ph)
                        if tf_util.is_image(self.observation_space):
                            tf.summary.image('observation', train_model.obs_ph)
                        else:
                            tf.summary.histogram(
                                'observation', train_model.obs_ph)
                self.train_model = train_model
                self.act_model = act_model
                self.step = act_model.step
                self.proba_step = act_model.proba_step
                self.value = act_model.value
                self.initial_state = act_model.initial_state
                tf.global_variables_initializer().run(
                    session=self.sess)  # pylint: disable=E1101

                self.summary = tf.summary.merge_all()

    def _train_step(
            self, learning_rate, srl_lr, cliprange, obs, next_obs, returns, true_reward, masks, actions,
            values, neglogpacs, update, writer, states=None, cliprange_vf=None):
        """
        Training of POAR Algorithm
        :param learning_rate: (float) learning rate
        :param srl_lr: (float) learning rate for the srl part
        :param cliprange: (float) Clipping factor
        :param obs: (np.ndarray) The current observation of the environment
        :param next_obs: successive observation
        :param returns: (np.ndarray) the rewards
        :param masks: (np.ndarray) The last masks for done episodes (used in recurrent policies)
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
            self.train_model.obs_ph: obs,
            self.action_ph: actions,
            self.true_reward_ph: true_reward,
            self.advs_ph: advs,
            self.rewards_ph: returns,
            self.learning_rate_ph: learning_rate,
            self.clip_range_ph: cliprange,
            self.old_neglog_pac_ph: neglogpacs,
            self.old_vpred_ph: values,
            self.srl_lr_ph: srl_lr
        }
        if states is not None:
            td_map[self.train_model.states_ph] = states
            td_map[self.train_model.dones_ph] = masks

        # Running for srl model
        if next_obs is not None:
            td_map[self.train_model.next_obs_ph] = next_obs
            td_map[self.train_model.action_ph] = actions

        if cliprange_vf is not None and cliprange_vf >= 0:
            td_map[self.clip_range_vf_ph] = cliprange_vf

        if states is None:
            update_fac = self.n_batch // self.nminibatches // self.noptepochs + 1
        else:
            update_fac = self.n_batch // self.nminibatches // self.noptepochs // self.n_steps + 1

        running_list = [
            self._train,
            self._srl_train,
            self.pg_loss,
            self.vf_loss,
            self.entropy,
            self.approxkl,
            self.clipfrac] + self.srl_loss_list
        if writer is not None:
            running_list.append(self.summary)
            # run loss backprop with summary, but once every 10 runs save the
            # metadata (memory, compute time, ...)
            if self.full_tensorboard_log and (1 + update) % 10 == 0:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                running_res = self.sess.run(running_list, td_map, options=run_options, run_metadata=run_metadata)
                writer.add_run_metadata(run_metadata, 'step%d' % (update * update_fac))
            else:
                running_res = self.sess.run(running_list, td_map)
            summary = running_res[-1]
            loss_vals = running_res[2:-1]
            writer.add_summary(summary, (update * update_fac))
        else:
            loss_vals = self.sess.run(running_list, td_map)[2:]
        return loss_vals

    def learn(self, total_timesteps, callback=None, seed=None, log_interval=1, tb_log_name="POAR", 
              reset_num_timesteps=True):
        # Transform to callable if needed
        self.learning_rate = get_schedule_fn(self.learning_rate)
        self.cliprange = get_schedule_fn(self.cliprange)
        self.srl_lr = get_schedule_fn(self.srl_lr)
        cliprange_vf = get_schedule_fn(self.cliprange_vf)
        new_tb_log = self._init_num_timesteps(reset_num_timesteps)

        with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name, new_tb_log) \
                as writer:
            self._setup_learn(seed)

            runner = Runner(env=self.env, model=self, n_steps=self.n_steps, gamma=self.gamma, lam=self.lam)
            self.episode_reward = np.zeros((self.n_envs,))

            ep_info_buf = deque(maxlen=100)
            t_first_start = time.time()
            batch_latent = []
            batch_reward = []
            n_updates = total_timesteps // self.n_batch
            for update in range(1, n_updates + 1):
                assert self.n_batch % self.nminibatches == 0

                batch_size = self.n_batch // self.nminibatches
                t_start = time.time()
                frac = 1.0 - (update - 1.0) / n_updates
                lr_now = self.learning_rate(frac)
                srl_lr_now = max(self.srl_lr(np.exp(-(update - 1.0) * 20  / n_updates)), 0.001*lr_now)
                cliprange_now = self.cliprange(frac)
                cliprange_vf_now = cliprange_vf(frac)
                # true_reward is the reward without discount
                # mb_obs, mb_ae_obs, mb_returns, mb_dones, mb_actions
                obs, next_obs, returns, masks, actions, values, neglogpacs, states, ep_infos, true_reward, \
                    reconstruct_image, latent, real_image = runner.run()


                self.num_timesteps += self.n_batch
                #print('n_batch', self.n_batch)
                #print('n_envs', self.n_envs)
                #print('n_steps', self.n_steps)
                #print('shape of obs', np.shape(obs))
                #print('shape of actions', np.shape(actions))
                #print("SRLSSSTATE_S", self.train_model.srl_state)

                ep_info_buf.extend(ep_infos)
                mb_loss_vals = []
                if states is None:  # nonrecurrent version
                    update_fac = self.n_batch // self.nminibatches // self.noptepochs + 1
                    inds = np.arange(self.n_batch)
                    for epoch_num in range(self.noptepochs):
                        np.random.shuffle(inds)
                        for start in range(0, self.n_batch, batch_size):
                            timestep = self.num_timesteps // update_fac + \
                                ((self.noptepochs * self.n_batch + epoch_num * self.n_batch + start) // batch_size)
                            end = start + batch_size
                            mbinds = inds[start:end]
                            slices = (
                                arr[mbinds] for arr in (obs, next_obs, returns, true_reward, masks, actions, values,
                                                        neglogpacs))
                            mb_loss_vals.append(self._train_step(lr_now, srl_lr_now, cliprange_now,
                                                                 *slices, writer=writer,
                                                                 update=timestep, cliprange_vf=cliprange_vf_now))

                loss_vals = np.mean(mb_loss_vals, axis=0)
                t_now = time.time()
                fps = int(self.n_batch / (t_now - t_start))

                if writer is not None:
                    self.episode_reward = total_episode_reward_logger(
                        self.episode_reward, true_reward.reshape(
                            (self.n_envs, self.n_steps)), masks.reshape(
                            (self.n_envs, self.n_steps)), writer, self.num_timesteps)
                    print(self.episode_reward, 0)


                    ###

                batch_latent.append(latent)

                batch_reward.append([-1 if r < 0 else r for r in true_reward])

                if update % 1000 == 0:#(n_updates // 100) 52

                    latent = np.concatenate(batch_latent)
                    batch_reward = np.concatenate(batch_reward)
                    # batch_reward = (batch_reward - np.min(batch_reward)) / (np.min(batch_reward) - np.max(batch_reward)) +1
                    # fig, ax = plt.subplots(nrows=1, ncols=2)
                    latent_pca = pca(latent, dim=2)
                    # This is for the cricular tasks!!!!!!!!!!!!!!!!!!!!!!!!!
                    #latent_pca = np.matmul(latent_pca, np.array([[np.cos(np.pi/4), -np.sin(np.pi/4)],[np.sin(np.pi/4), np.cos(np.pi/4)]]))
                    # zeros = latent_pca[np.where(abs(batch_reward-3.5) < 3.5)].T
                    # positive = latent_pca[np.where(batch_reward > 7)].T
                    # negative = latent_pca[np.where(batch_reward < 0)].T
                    # zeros = latent_pca[np.where(batch_reward == 0)].T
                    # positive = latent_pca[np.where(batch_reward > 0)].T
                    # negative = latent_pca[np.where(batch_reward < 0)].T
                    # ax[0].imshow((reconstruct_image[0] + 1) / 2)
                    # ax[1].scatter(zeros[0], zeros[1], c='y', s=5, label='null')
                    # ax[1].scatter(negative[0], negative[1], c='b', s=3, label='-')
                    # ax[1].scatter(positive[0], positive[1], c='r', s=3, label='+')
                    # ax[1].legend()

                    fig = plt.figure(figsize=(8,8));
                    sc = plt.scatter(latent_pca[:,0], latent_pca[:,1], s=4, c = batch_reward, cmap=plt.cm.get_cmap('Spectral_r'));
                    plt.colorbar(sc)
                    # plt.scatter(zeros[0], zeros[1], c='y', s=4, label='null')
                    # plt.scatter(negative[0], negative[1], c='b', s=3, label='-')
                    # plt.scatter(positive[0], positive[1], c='r', s=4, label='+')

                    #sc = plt.scatter(latent_pca[:,0], latent_pca[:,1], s=4, c = batch_reward, cmap=plt.cm.get_cmap('Spectral_r'))
                    #plt.colorbar(sc)

                    # plt.legend()

                    # fig = plt.figure(figsize=(10, 10));plt.scatter(zeros[0], zeros[1], c='y', s=3, label='null');plt.scatter(negative[0], negative[1], c='b', s=3, label='-');plt.scatter(positive[0], positive[1], c='r', s=3, label='+');plt.legend();plt.show()
                    plt.savefig("/home/tete/work/robotics-rl-srl/Reconstruction/state_{}".format(update) + ".png")

                    if np.mean(self.episode_reward) > 1800:

                        from mpl_toolkits.mplot3d import Axes3D
                        latent_pca = pca(latent, dim=3).T
                        tt()
                        fig = plt.figure()
                        ax = Axes3D(fig)
                        sc = ax.scatter(latent_pca[0], latent_pca[1], latent_pca[2], c=batch_reward, cmap=plt.cm.get_cmap('Spectral_r'))
                        plt.colorbar(sc)
                        fig=plt.figure(); ax=Axes3D(fig);sc = ax.scatter(latent_pca[0], latent_pca[1], latent_pca[2], c=batch_reward, s=4, cmap=plt.cm.get_cmap('Spectral_r'));plt.colorbar(sc);plt.show()

                    batch_reward, batch_latent = [], []
                    plt.close(fig)


                        ###



                if self.verbose >= 1 and (
                        update % log_interval == 0 or update == 1):
                    explained_var = explained_variance(values, returns)
                    logger.logkv("serial_timesteps", update * self.n_steps)
                    logger.logkv("n_updates", update)
                    logger.logkv("total_timesteps", self.num_timesteps)
                    logger.logkv("fps", fps)
                    logger.logkv("explained_variance", float(explained_var))
                    if len(ep_info_buf) > 0 and len(ep_info_buf[0]) > 0:
                        logger.logkv('ep_reward_mean', safe_mean(
                            [ep_info['r'] for ep_info in ep_info_buf]))
                        logger.logkv('ep_len_mean', safe_mean(
                            [ep_info['l'] for ep_info in ep_info_buf]))
                    logger.logkv('time_elapsed', t_start - t_first_start)
                    for (
                            loss_val,
                            loss_name) in zip(
                            loss_vals,
                            self.loss_names):
                        logger.logkv(loss_name, loss_val)
                    logger.dumpkvs()

                if callback is not None:
                    # Only stop training if return value is False, not when it is None. This is for backwards
                    # compatibility with callbacks that have no return
                    # statement.
                    if callback(locals(), globals()) is False:
                        break

            return self

    def save(self, save_path):
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
            "_vectorize_action": self._vectorize_action,
            "policy_kwargs": self.policy_kwargs
        }

        params_to_save = self.get_parameters()

        self._save_to_file(save_path, data=data, params=params_to_save)


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
        mb_latent = []
        mb_next_obs = []
        mb_states = self.states
        ep_infos = []
        iteration = 0
        ae_obs = 0
        while iteration < self.n_steps:
            actions, values, self.states, neglogpacs, ae_obs, latent = self.model.step(
                self.obs, self.states, self.dones)
            mb_latent.append(latent)
            mb_obs.append(self.obs.copy())  # 每次添加num_cpu个 128*num_cpu (n_env)
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.env.action_space, gym.spaces.Box):
                clipped_actions = np.clip(
                    actions,
                    self.env.action_space.low,
                    self.env.action_space.high)
            self.obs[:], rewards, self.dones, infos = self.env.step(
                clipped_actions)
            if True in self.dones:
                mb_latent.pop()
                mb_obs.pop()
                mb_actions.pop()
                mb_values.pop()
                mb_neglogpacs.pop()
                mb_dones.pop()
                continue
            mb_next_obs.append(self.obs.copy())
            for info in infos:
                maybe_ep_info = info.get('episode')
                if maybe_ep_info is not None:
                    ep_infos.append(maybe_ep_info)
            mb_rewards.append(rewards)
            iteration += 1

        # batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_latent = np.asarray(mb_latent, dtype=np.float32)
        mb_next_obs = np.asarray(mb_next_obs, dtype=self.obs.dtype)
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
            delta = mb_rewards[step] + self.gamma * \
                nextvalues * nextnonterminal - mb_values[step]
            mb_advs[step] = last_gae_lam = delta + self.gamma * \
                self.lam * nextnonterminal * last_gae_lam
        mb_returns = mb_advs + mb_values
        # before this map, the mb_obs has dimension [num_step, n_env, (image_shape)]
        # for the second dimension, the sequential is continuous
        mb_obs, mb_next_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, true_reward, mb_latent = \
            map(swap_and_flatten, (mb_obs, mb_next_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs,
                                   true_reward, mb_latent))
        return (mb_obs, mb_next_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, mb_states, ep_infos,
                true_reward, ae_obs, mb_latent, self.obs)


def get_schedule_fn(value_schedule):
    """
    Transform (if needed) learning rate and clip range
    to callable.

    :param value_schedule: (callable or float)
    :return: (function)
    """
    # If the passed schedule is a float
    # create a constant function
    if isinstance(value_schedule, (float, int)):
        # Cast to float to avoid errors
        value_schedule = constfn(float(value_schedule))
    else:
        assert callable(value_schedule)
    return value_schedule


def swap_and_flatten(arr):
    """
    swap and then flatten axes 0 and 1

    :param arr: (np.ndarray)
    :return: (np.ndarray)
    """
    shape = arr.shape
    return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])


def constfn(val):
    """
    Create a function that returns a constant
    It is useful for learning rate schedule (to avoid code duplication)

    :param val: (float)
    :return: (function)
    """

    def func(_):
        return val

    return func


def safe_mean(arr):
    """
    Compute the mean of an array if there is at least one element.
    For empty array, return nan. It is used for logging only.

    :param arr: (np.ndarray)
    :return: (float)
    """
    return np.nan if len(arr) == 0 else np.mean(arr)
