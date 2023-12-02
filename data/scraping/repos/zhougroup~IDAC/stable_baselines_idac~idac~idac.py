##0323
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
from stable_baselines.sac.replay_buffer import ReplayBuffer
from stable_baselines.ppo2.ppo2 import safe_mean, get_schedule_fn
from stable_baselines.sac.idac_policy import SACPolicy
from stable_baselines import logger

import tensorflow_probability as tfp


EPS = 1e-6 
def get_vars(scope):
    """
    Alias for get_trainable_vars

    :param scope: (str)
    :return: [tf Variable]
    """
    return tf_util.get_trainable_vars(scope)


class IDAC(OffPolicyRLModel):
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
                 learning_starts=100, train_freq=1, batch_size=64, kl_term=1.0, max_ep_length=1000,
                 tau=0.005, ent_coef='auto', target_update_interval=1,noise_dim=5,noise_num=1,J=51,L=21,K=51,
                 gradient_steps=1, target_entropy='auto', action_noise=None,
                 random_exploration=0.0, verbose=0, tensorboard_log=None,
                 _init_setup_model=True, policy_kwargs=None, full_tensorboard_log=False):

        super(IDAC, self).__init__(policy=policy, env=env, replay_buffer=None, verbose=verbose,
                                  policy_base=SACPolicy, requires_vec_env=False, policy_kwargs=policy_kwargs)

        self.buffer_size = buffer_size
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
        self.action_target_ph = None
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
        self.noise_dim = noise_dim
        self.noise_num = noise_num
        self.kl_term = kl_term
        self.max_ep_length = max_ep_length
        self.J = J
        self.L = L 
        self.K = K

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

                self.replay_buffer = ReplayBuffer(self.buffer_size)

                with tf.variable_scope("input", reuse=False):
                    # Create policy and target TF objects n_env=1, n_steps=1, n_batch=None, noise_dim=100
                    self.policy_tf = self.policy(self.sess, self.observation_space, self.action_space, noise_dim=self.noise_dim,
                                                 **self.policy_kwargs)
                    self.target_policy = self.policy(self.sess, self.observation_space, self.action_space, noise_dim=self.noise_dim,
                                                     **self.policy_kwargs)

                    # Initialize Placeholders
                    self.observations_ph = self.policy_tf.obs_ph #check
                    # Normalized observation for pixels
                    self.processed_obs_ph = self.policy_tf.processed_obs
                    self.next_observations_ph = self.target_policy.obs_ph #check
                    self.processed_next_obs_ph = self.target_policy.processed_obs
                    
                    self.action_target_ph = tf.placeholder(tf.float32, shape=(None,) + self.action_space.shape, name='actions_target_ph')
                    self.actions_ph = tf.placeholder(tf.float32, shape=(None,) + self.action_space.shape,
                                                     name='actions') #check

                    self.terminals_ph = tf.placeholder(tf.float32, shape=(None, 1), name='terminals')
                    self.rewards_ph = tf.placeholder(tf.float32, shape=(None, 1), name='rewards')
                    self.learning_rate_ph = tf.placeholder(tf.float32, [], name="learning_rate_ph")

                    self.idx_ph = tf.placeholder(tf.int32,[None], name="idx_ph")
                    self.idx_ph_L = tf.placeholder(tf.int32,[None], name="idx_ph_L")
                    self.idx_ph_J = tf.placeholder(tf.int32,[None], name="idx_ph_J")
                    self.idx_ph_K = tf.placeholder(tf.int32,[None], name="idx_ph_K")
                    self.noises_ph = self.policy_tf.noise_ph
                    self.next_noises_ph = self.target_policy.noise_ph
                    self.noises_ph_plus = tf.placeholder(tf.float32, shape=(None,self.noise_dim), name='noises_ph_plus')
                    self.next_noises_ph_plus = tf.placeholder(tf.float32, shape=(None,self.noise_dim), name='next_noises_ph_plus')                    

                with tf.variable_scope("model", reuse=False):
                    # Create the policy
                    # first return value corresponds to deterministic actions
                    # policy_out corresponds to stochastic actions, used for training
                    # logp_pi is the log probabilty of actions taken by the policy
                    
                    noises_out = self.policy_tf.make_noise(self.processed_obs_ph)

                    deterministic_action, policy_out_, _, mu_act, logstd_act, act_ori = self.policy_tf.make_actor(self.processed_obs_ph, self.noises_ph)

                    policy_out = tf.reshape(tf.gather(policy_out_, self.idx_ph_J), (self.batch_size, self.J, 1, self.action_space.shape[0]))
                    act_ori = tf.reshape(tf.gather(act_ori, self.idx_ph_J), (self.batch_size, self.J, 1, self.action_space.shape[0]))
                    logstd_act = tf.stop_gradient(logstd_act)
                    mu_act = tf.stop_gradient(mu_act)

                    mu_act_generate_actions = tf.reshape(tf.gather(mu_act, self.idx_ph_J),(self.batch_size, self.J, 1, self.action_space.shape[0]))
                    logstd_act_generate_actions = tf.reshape(tf.gather(logstd_act, self.idx_ph_J),(self.batch_size, self.J, 1, self.action_space.shape[0]))

                    mu_act_not_generate_actions = tf.reshape(tf.gather(mu_act, self.idx_ph_L), (self.batch_size, 1, self.L, self.action_space.shape[0]))
                    logstd_act_not_generate_actions = tf.reshape(tf.gather(logstd_act, self.idx_ph_L), (self.batch_size, 1, self.L, self.action_space.shape[0]))
 

                    logp_pi_sivi_partI = tf.reduce_sum(-0.5 * (((act_ori - mu_act_generate_actions) /
                         (tf.exp(logstd_act_generate_actions) + EPS)) ** 2 + 2 * logstd_act_generate_actions + np.log(2 * np.pi)),axis=-1, keep_dims=True)
                    
                    logp_pi_sivi_partI -= tf.reduce_sum(tf.log(1 - policy_out ** 2 + EPS), axis=-1, keep_dims=True)

                    logp_pi_sivi_partI = tf.squeeze(logp_pi_sivi_partI, axis = -1)


                    logp_pi_sivi_partII = tf.reduce_sum(-0.5 * (((act_ori - mu_act_not_generate_actions) /
                         (tf.exp(logstd_act_not_generate_actions) + EPS)) ** 2 + 2 * logstd_act_not_generate_actions + np.log(2 * np.pi)),axis=-1, keep_dims=True)

                    logp_pi_sivi_partII -= tf.reduce_sum(tf.log(1 - policy_out ** 2 + EPS), axis=-1, keep_dims=True)

                    logp_pi_sivi_partII = tf.squeeze(logp_pi_sivi_partII, axis = -1)

                    logp_pi_sivi = tf.concat([logp_pi_sivi_partI, logp_pi_sivi_partII], axis = -1)
                    
                    logp_pi_sivi = tf.math.reduce_logsumexp(logp_pi_sivi, axis=-1) - tf.math.log(tf.cast(self.L+1, tf.float32))

                    logp_pi_sivi = tf.reduce_mean(logp_pi_sivi, axis=1, keep_dims = True)
                    self.deterministic_action = deterministic_action

                    self.entropy = tf.reduce_mean(self.policy_tf.entropy)

                    #  Use two Q-functions to improve performance by reducing overestimation bias.                    
                    qf1, _, _ = self.policy_tf.make_critics(tf.gather(self.processed_obs_ph, self.idx_ph_K),  self.actions_ph, tf.gather(self.noises_ph, self.idx_ph_K),
                                                                     create_qf=True, create_vf=False)

                    _, qf2, _ = self.policy_tf.make_critics(tf.gather(self.processed_obs_ph, self.idx_ph_K),  self.actions_ph, tf.gather(self.noises_ph_plus, self.idx_ph_K),
                                                                     create_qf=True, create_vf=False,reuse=True)


                    self.qf1 = qf1

                    qf1_sort = tf.sort(tf.reshape(qf1, (-1, self.noise_num)), axis=1, direction='ASCENDING')
                    qf2_sort = tf.sort(tf.reshape(qf2, (-1, self.noise_num)), axis=1, direction='ASCENDING')


                    qf1_pi, qf2_pi, _ = self.policy_tf.make_critics(tf.gather(self.processed_obs_ph,self.idx_ph_J),
                                                                    tf.gather(policy_out_, self.idx_ph_J),
                                                                    tf.gather(noises_out, self.idx_ph_J), 
                                                                    create_qf=True, create_vf=False,
                                                                    reuse=True)
                    # first average over K
                    qf1_pi = tf.reduce_mean(tf.reshape(qf1_pi,(-1,self.J)),axis=1, keep_dims = True)
                    qf2_pi = tf.reduce_mean(tf.reshape(qf2_pi,(-1,self.J)),axis=1, keep_dims = True)

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
                    
                    qf1_target, _, _ = self.target_policy.make_critics(self.processed_next_obs_ph, self.action_target_ph, self.next_noises_ph,
                                                                         create_qf=True, create_vf=False)
                    _, qf2_target, _ = self.target_policy.make_critics(self.processed_next_obs_ph, self.action_target_ph, self.next_noises_ph_plus,
                                                                         create_qf=True, create_vf=False,reuse=True)                    

                    self.qf1_target = qf1_target


                with tf.variable_scope("loss", reuse=False):
                    # Take the min of the two Q-Values (Double-Q Learning)
                    mean_qf_pi = (qf1_pi + qf2_pi) / 2.0

                    # Target for Q value regression
                    qf1_backup = tf.stop_gradient(
                        self.rewards_ph + 
                        (1 - self.terminals_ph) * self.gamma * qf1_target
                    )
                    qf2_backup = tf.stop_gradient(
                        self.rewards_ph + 
                        (1 - self.terminals_ph) * self.gamma * qf2_target
                    )                    
                    
                    qf1_backup = tf.sort(tf.reshape(qf1_backup, (-1, self.noise_num)),axis=1,  direction='ASCENDING' )
                    qf2_backup = tf.sort(tf.reshape(qf2_backup, (-1, self.noise_num)),axis=1,  direction='ASCENDING' )

                    qf_backup = tf.minimum(qf1_backup, qf2_backup)
                    
                    qf_backup = tf.tile(tf.expand_dims(qf_backup, axis=2), [1, 1, self.noise_num])
                    qf1_sort = tf.tile(tf.expand_dims(qf1_sort, axis=1), [1, self.noise_num, 1])
                    qf2_sort = tf.tile(tf.expand_dims(qf2_sort, axis=1), [1, self.noise_num, 1])

                    error_loss = qf_backup - qf1_sort
                    Huber_loss = tf.losses.huber_loss(qf_backup, qf1_sort, reduction = tf.losses.Reduction.NONE)
                    min_tau = 1/(2*self.noise_num)
                    max_tau = (2*self.noise_num+1)/(2*self.noise_num)
                    tau = tf.reshape (tf.range(min_tau, max_tau, 1/self.noise_num), [1, self.noise_num])
                    inv_tau = 1.0 - tau 
                    qf1_loss_tmp = tf.where(tf.less(error_loss, 0.0), inv_tau * Huber_loss, tau * Huber_loss)
                    qf1_loss = tf.reduce_mean(tf.reduce_sum(tf.reduce_mean(qf1_loss_tmp, axis = 2), axis = 1))
                    
                    error_loss = qf_backup - qf2_sort
                    Huber_loss = tf.losses.huber_loss(qf_backup, qf2_sort, reduction = tf.losses.Reduction.NONE)
                    min_tau = 1/(2*self.noise_num)
                    max_tau = (2*self.noise_num+1)/(2*self.noise_num)
                    tau = tf.reshape (tf.range(min_tau, max_tau, 1/self.noise_num), [1, self.noise_num])
                    inv_tau = 1.0 - tau 
                    qf2_loss_tmp = tf.where(tf.less(error_loss, 0.0), inv_tau * Huber_loss, tau * Huber_loss)
                    qf2_loss = tf.reduce_mean(tf.reduce_sum(tf.reduce_mean(qf2_loss_tmp, axis = 2), axis = 1))


                    # Compute the entropy temperature loss
                    # it is used when the entropy coefficient is learned
                    ent_coef_loss, entropy_optimizer = None, None
                    if not isinstance(self.ent_coef, float):
                        ent_coef_loss = -tf.reduce_mean(
                            self.log_ent_coef * tf.stop_gradient(logp_pi_sivi + self.target_entropy))
                        entropy_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph)

                    # Compute the policy loss

                    policy_loss = tf.reduce_mean(self.ent_coef * logp_pi_sivi - mean_qf_pi)

                    # Target for value fn regression
                    # We update the vf towards the min of two Q-functions in order to
                    # reduce overestimation bias from function approximation error.

                    values_losses = qf1_loss + qf2_loss

                    # Policy train op
                    # (has to be separate from value train op, because min_qf_pi appears in policy_loss)
                    policy_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph)
                    policy_params = get_vars('model/pi')+get_vars('model/inf')
                    policy_train_op = policy_optimizer.minimize(policy_loss, var_list=policy_params)

                    # Value train op
                    value_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph)
                    values_params = get_vars('model/values_fn')

                    source_params = get_vars("model/values_fn/qf")
                    target_params = get_vars("target/values_fn/qf")

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
                        self.train_values_op=train_values_op = value_optimizer.minimize(values_losses, var_list=values_params)

                        self.infos_names = ['policy_loss', 'qf1_loss', 'qf2_loss', 'entropy']
                        # All ops to call during one training step
                        self.step_ops = [policy_loss, qf1_loss, qf2_loss,
                                         qf1, qf2, logp_pi_sivi,
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
                    tf.summary.scalar('entropy', self.entropy)
                    if ent_coef_loss is not None:
                        tf.summary.scalar('ent_coef_loss', ent_coef_loss)
                        tf.summary.scalar('ent_coef', self.ent_coef)

                    tf.summary.scalar('learning_rate', tf.reduce_mean(self.learning_rate_ph))

                # Retrieve parameters that must be saved
                self.params = get_vars("model")
                self.target_params = get_vars("target/values_fn/qf")

                # Initialize Variables and target network
                with self.sess.as_default():
                    self.sess.run(tf.global_variables_initializer())
                    self.sess.run(target_init_op)

                self.summary = tf.summary.merge_all()

    def _train_step(self, step, writer, learning_rate, dis_eval_array, dis_eval_interval, dis_path):
        # Sample a batch from the replay buffer
        batch = self.replay_buffer.sample(self.batch_size)
        batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones, batch_noises = batch


        idx = np.arange(0, self.batch_size*(self.L+self.J), self.L+self.J)
        n_ac = batch_actions.shape[1]
        n_s = batch_obs.shape[1]
        a_tile = np.reshape(np.tile(batch_actions, [1, self.noise_num]), (-1,n_ac)) #because L*J is usually bigger than numofnoise
        obs_tile = np.reshape(np.tile(batch_obs, [1, self.L+self.J]), (-1,n_s)) 
        next_obs_tile = np.reshape(np.tile(batch_next_obs, [1, self.noise_num]), (-1,n_s))

        batch_noise_tile = self.policy_tf.gen_noise(obs_tile)
        batch_noise_tile[idx,:] = batch_noises

        batch_next_noise_tile = self.policy_tf.gen_noise(next_obs_tile)
        batch_next_noise = batch_next_noise_tile[np.arange(0, self.batch_size*(self.noise_num), self.noise_num),:]
        batch_next_actions_tile = self.policy_tf.step(batch_next_obs,batch_next_noise)
        batch_next_actions_tile = np.reshape(np.tile(batch_next_actions_tile, [1, self.noise_num]), (-1,n_ac))


        reward_rescale = batch_rewards.reshape(self.batch_size, -1) 
        reward_tile = np.reshape(np.tile(reward_rescale, [1, self.noise_num]), (-1,1))
        terminals = batch_dones.reshape(self.batch_size, -1)
        terminals_tile = np.reshape(np.tile(terminals, [1, self.noise_num]), (-1,1))

        idx_K = np.copy(idx)
        idx_J = np.copy(idx)

        for i_tmp in range(1,self.J):
            new_idx = np.arange(i_tmp, self.batch_size*(self.L+self.J), self.L+self.J)
            idx_J = np.concatenate([idx_J, new_idx])

        idx_J = np.sort(idx_J)

        idx_L = np.setdiff1d(np.arange(0, self.batch_size*(self.L+self.J)), idx_J)

        for i_tmp in range(1,self.noise_num):
            new_idx = np.arange(i_tmp, self.batch_size*(self.L+self.J), (self.L+self.J))
            idx_K = np.concatenate([idx_K, new_idx])

        idx_K = np.sort(idx_K)
        
        feed_dict = {
            self.observations_ph: obs_tile,
            self.actions_ph: a_tile,
            self.next_observations_ph: next_obs_tile,
            self.rewards_ph: reward_tile,
            self.terminals_ph: terminals_tile,
            self.learning_rate_ph: learning_rate,
            self.idx_ph: idx,
            self.noises_ph: batch_noise_tile,
            self.next_noises_ph: batch_next_noise_tile,
            self.action_target_ph: batch_next_actions_tile,
            self.idx_ph_J: idx_J,
            self.idx_ph_K: idx_K,
            self.idx_ph_L: idx_L,
            self.next_noises_ph_plus: self.policy_tf.gen_noise(next_obs_tile),
            self.noises_ph_plus: self.policy_tf.gen_noise(obs_tile),                        
        }
        
        
        ## save intermediate output for distributional matching
        if step % dis_eval_interval == 0:
            n_batch = 100
            s0 = batch_obs[0]
            a0 = batch_actions[0]
            r0 = batch_rewards[0]
            gamma = self.gamma
            s1 = batch_next_obs[0]

            s0_batch = np.tile(s0[None], [n_batch, 1])
            noise_0_batch = self.policy_tf.gen_noise(s0_batch)
            a0_batch = np.tile(a0[None], [n_batch, 1])

            s1_batch = np.tile(s1[None], [n_batch, 1])
            noise_1_batch = self.policy_tf.gen_noise(s1_batch)  # for K=1
            a1_batch = self.policy_tf.step(s1_batch, noise_1_batch)

            q1_target = self.sess.run(self.qf1_target, feed_dict={self.next_observations_ph: s1_batch,
                                                                  self.action_target_ph: a1_batch,
                                                                  self.next_noises_ph: noise_1_batch,
                                                                  self.noises_ph: noise_0_batch,
                                                                  })

            g0 = q1_target

            g1 = r0 + gamma * q1_target

            g = np.concatenate((g0, g1), axis=0)
            dis_eval_array.append(g)

            eval_array = np.array(dis_eval_array)
            np.save(dis_path, eval_array)


        # Do one gradient step
        # and optionally compute log for tensorboard
        if writer is not None:
            out = self.sess.run([self.summary] + self.step_ops, feed_dict)
            summary = out.pop(0)
            writer.add_summary(summary, step)
        else:
            out = self.sess.run(self.step_ops, feed_dict)

        # Unpack to monitor losses and entropy
        policy_loss, qf1_loss, qf2_loss, *values = out
        entropy = values[3]


        if self.log_ent_coef is not None:
            ent_coef_loss, ent_coef = values[-2:]
            return policy_loss, qf1_loss, qf2_loss, entropy, ent_coef_loss, ent_coef

        return policy_loss, qf1_loss, qf2_loss, entropy

    def learn(self, total_timesteps, env_eval, callback=None, seed=None, path=None, dis_path=None, score_path=None,
              dis_eval_interval=100, log_interval=4, tb_log_name="SAC", reset_num_timesteps=True, replay_wrapper=None):

        self.eval_env = env_eval
        new_tb_log = self._init_num_timesteps(reset_num_timesteps)

        if replay_wrapper is not None:
            self.replay_buffer = replay_wrapper(self.replay_buffer)

        with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name, new_tb_log) \
                as writer:

            self._setup_learn(seed)

            # Transform to callable if needed
            self.learning_rate = get_schedule_fn(self.learning_rate)
            # Initial learning rate
            current_lr = self.learning_rate(1)

            start_time = time.time()
            episode_rewards = [0.0]
            episode_successes = []
            if self.action_noise is not None:
                self.action_noise.reset()
            obs = self.env.reset()
            self.episode_reward = np.zeros((1,))
            ep_info_buf = deque(maxlen=100)
            n_updates = 0
            infos_values = []

            dis_eval_array = []  # (total_step % eval_intervel) x 2 x n_batch
            self.ep_length = 0

            for step in range(total_timesteps):
                if callback is not None:
                    # Only stop training if return value is False, not when it is None. This is for backwards
                    # compatibility with callbacks that have no return statement.
                    if callback(locals(), globals()) is False:
                        break

                # Before training starts, randomly sample actions
                # from a uniform distribution for better exploration.
                # Afterwards, use the learned policy
                # if random_exploration is set to 0 (normal setting)
                if (self.num_timesteps < self.learning_starts
                    or np.random.rand() < self.random_exploration):
                    # No need to rescale when sampling random action
                    rescaled_action = action = self.env.action_space.sample()
                    noise = np.zeros(self.noise_dim)
                else:            
                    noise = self.policy_tf.gen_noise(obs[None]).flatten()
                    action = self.policy_tf.step(obs[None],noise[None] ,deterministic=False).flatten()

                    # Add noise to the action (improve exploration,
                    # not needed in general)
                    if self.action_noise is not None:
                        action = np.clip(action + self.action_noise(), -1, 1)
                    # Rescale from [-1, 1] to the correct bounds
                    rescaled_action = action * np.abs(self.action_space.low)

                assert action.shape == self.env.action_space.shape

                new_obs, reward, done, info = self.env.step(rescaled_action)
                self.ep_length += 1

                # Store transition in the replay buffer.
                self.replay_buffer.add(obs, action, reward, new_obs, float(done), noise)

                episode_rewards[-1] += reward
                reset_flag = done or self.ep_length >= self.max_ep_length
                if reset_flag:
                    if self.action_noise is not None:
                        self.action_noise.reset()
                    obs = self.env.reset()
                    episode_rewards.append(0.0)
                    self.ep_length = 0

                    maybe_is_success = info.get('is_success')
                    if maybe_is_success is not None:
                        episode_successes.append(float(maybe_is_success))
                else:
                    obs = new_obs

                # Retrieve reward and episode length if using Monitor wrapper
                maybe_ep_info = info.get('episode')
                if maybe_ep_info is not None:
                    ep_info_buf.extend([maybe_ep_info])

                if writer is not None:
                    # Write reward per episode to tensorboard
                    ep_reward = np.array([reward]).reshape((1, -1))
                    ep_done = np.array([done]).reshape((1, -1))
                    self.episode_reward = total_episode_reward_logger(self.episode_reward, ep_reward,
                                                                      ep_done, writer, self.num_timesteps)

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

                        mb_infos_vals.append(self._train_step(step, writer, current_lr, dis_eval_array,
                                                              dis_eval_interval, dis_path))
                        # Update target network
                        if (step + grad_step) % self.target_update_interval == 0:
                            # Update target network
                            self.sess.run(self.target_update_op)
                    # Log losses and entropy, useful for monitor training
                    if len(mb_infos_vals) > 0:
                        infos_values = np.mean(mb_infos_vals, axis=0)

                if self.num_timesteps % 2000 == 0:
                    eval_ob = self.eval_env.reset()
                    eval_epi_rewards = 0
                    eval_epis = 0
                    eval_performance = []
                    eval_ep_step = 0
                    while True:
                        eval_noise = self.policy_tf.gen_noise(eval_ob[None]).flatten()
                        eval_action = self.policy_tf.step(eval_ob[None], eval_noise[None], deterministic=True).flatten()
                        eval_rescaled_action = eval_action * np.abs(self.action_space.low)
                        eval_new_obs, eval_reward, eval_done, eval_info = self.eval_env.step(eval_rescaled_action)
                        eval_epi_rewards += eval_reward
                        eval_ob = eval_new_obs
                        eval_ep_step += 1
                        if eval_done or eval_ep_step >= self.max_ep_length:
                            eval_ob = self.eval_env.reset()
                            eval_performance.append(eval_epi_rewards)
                            eval_epi_rewards = 0
                            eval_epis += 1
                            eval_ep_step = 0
                            if eval_epis > 5:
                                break
                    with open(score_path, 'a') as f2:
                        f2.write("%i %f\n" % (self.num_timesteps, np.mean(eval_performance)))

                if len(episode_rewards[-101:-1]) == 0:
                    mean_reward = -np.inf
                else:
                    mean_reward = round(float(np.mean(episode_rewards[-101:-1])), 1)

                num_episodes = len(episode_rewards)
                self.num_timesteps += 1
                # Display training infos
                if self.verbose >= 1 and reset_flag and log_interval is not None and len(episode_rewards) % log_interval == 0:
                    fps = int(step / (time.time() - start_time))
                    logger.logkv("episodes", num_episodes)
                    logger.logkv("mean 100 episode reward", mean_reward)
                    with open(path,'a') as f1:
                        f1.write("%f " % step)
                        f1.write("%f " % mean_reward)
                    if len(ep_info_buf) > 0 and len(ep_info_buf[0]) > 0:
                        logger.logkv('ep_rewmean', safe_mean([ep_info['r'] for ep_info in ep_info_buf]))
                        logger.logkv('eplenmean', safe_mean([ep_info['l'] for ep_info in ep_info_buf]))
                        with open(path,'a') as f1:
                            f1.write("%f " % safe_mean([ep_info['r'] for ep_info in ep_info_buf]))
                            f1.write("%f " % safe_mean([ep_info['l'] for ep_info in ep_info_buf]))
                    logger.logkv("n_updates", n_updates)
                    with open(path,'a') as f1:
                        f1.write("%f " % n_updates)                    
                    logger.logkv("current_lr", current_lr)
                    logger.logkv("fps", fps)
                    logger.logkv('time_elapsed', int(time.time() - start_time))
                    
                    if len(episode_successes) > 0:
                        logger.logkv("success rate", np.mean(episode_successes[-100:]))
                    if len(infos_values) > 0:
                        for (name, val) in zip(self.infos_names, infos_values):
                            logger.logkv(name, val)

                    logger.logkv("total timesteps", self.num_timesteps)
                    logger.dumpkvs()
                    # Reset infos:
                    infos_values = []

            return self

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

    def evaluate(self, env_eval):
        obs = env_eval.reset()
        rew = 0
        while True:
            noise = self.policy_tf.gen_noise(obs[None]).flatten()
            action = self.policy_tf.step(obs[None], noise[None], deterministic=False).flatten()
            rescaled_action = action * np.abs(self.action_space.low)
            obs, reward, done, _ = env_eval.step(rescaled_action)
            rew += reward
            if done:
                break
        return rew

