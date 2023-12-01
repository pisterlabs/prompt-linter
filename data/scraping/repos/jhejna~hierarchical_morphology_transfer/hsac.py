import sys
import time
from collections import deque
import warnings

import numpy as np
import tensorflow as tf

from stable_baselines.a2c.utils import total_episode_reward_logger
from stable_baselines.common import tf_util, SetVerbosity, TensorboardWriter
from stable_baselines.common.vec_env import VecEnv
from stable_baselines.deepq.replay_buffer import ReplayBuffer
from stable_baselines.ppo2.ppo2 import safe_mean, get_schedule_fn
from stable_baselines.sac.policies import SACPolicy
from stable_baselines import logger

from bot_transfer.algs.hrl import OffPolicyHRLModel

def get_vars(scope):
    """
    Alias for get_trainable_vars

    :param scope: (str)
    :return: [tf Variable]
    """
    return tf_util.get_trainable_vars(scope)

def scale_action(action_space, action):
    """
    Rescale the action from [low, high] to [-1, 1]
    (no need for symmetric action space)
    :param action_space: (gym.spaces.box.Box)
    :param action: (np.ndarray)
    :return: (np.ndarray)
    """
    low, high = action_space.low, action_space.high
    return 2.0 * ((action - low) / (high - low)) - 1.0


def unscale_action(action_space, scaled_action):
    """
    Rescale the action from [-1, 1] to [low, high]
    (no need for symmetric action space)
    :param action_space: (gym.spaces.box.Box)
    :param action: (np.ndarray)
    :return: (np.ndarray)
    """
    low, high = action_space.low, action_space.high
    return low + (0.5 * (scaled_action + 1.0) * (high - low))


class HSAC(OffPolicyHRLModel):
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
    :param seed: (int) Seed for the pseudo-random generators (python, numpy, tensorflow).
        If None (default), use random seed. Note that if you want completely deterministic
        results, you must set `n_cpu_tf_sess` to 1.
    :param n_cpu_tf_sess: (int) The number of threads for TensorFlow operations
        If None, the number of cpu of the current machine will be used.
    """

    def __init__(self, low_policy, high_policy, env, 
                 low_gamma=0.99, high_gamma=0.99,
                 low_learning_rate=3e-4, high_learning_rate=3e-4,
                 low_buffer_size=50000, high_buffer_size=50000,
                 learning_starts=100,
                 low_train_freq=1, high_train_freq=1,
                 low_batch_size=64, high_batch_size=64,
                 low_tau=0.005, high_tau=0.005,
                 low_ent_coef='auto', high_ent_coef='auto',
                 low_target_update_interval=1, high_target_update_interval=1,
                 low_gradient_steps=1, high_gradient_steps=1,
                 low_target_entropy='auto', high_target_entropy='auto',
                 low_action_noise=None, high_action_noise=None,
                 low_random_exploration=0.0, high_random_exploration=0.0,
                 verbose=0, tensorboard_log=None,
                 _init_setup_model=True, 
                 low_policy_kwargs=None, high_policy_kwargs=None,
                 full_tensorboard_log=False,
                 seed=None, n_cpu_tf_sess=None):

        super(HSAC, self).__init__(low_policy=low_policy, high_policy=high_policy, env=env, low_replay_buffer=None, high_replay_buffer=None, verbose=verbose,
                                  low_policy_base=SACPolicy, high_policy_base=SACPolicy, requires_vec_env=False, low_policy_kwargs=low_policy_kwargs, high_policy_kwargs=high_policy_kwargs,
                                  seed=seed, n_cpu_tf_sess=n_cpu_tf_sess)

        print("LOW GRAD STEPS", low_gradient_steps)
        print("LOW POLICY ARGS", low_policy_kwargs)

        # Low Parameters
        self.low_buffer_size = low_buffer_size
        self.low_learning_rate = low_learning_rate
        self.low_train_freq = low_train_freq
        self.low_batch_size = low_batch_size
        self.low_tau = low_tau
        self.low_ent_coef = low_ent_coef
        self.low_target_update_interval = low_target_update_interval
        self.low_gradient_steps = low_gradient_steps
        self.low_gamma = low_gamma
        self.low_action_noise = low_action_noise
        self.low_random_exploration = low_random_exploration

        # High Parameters
        self.high_buffer_size = high_buffer_size
        self.high_learning_rate = high_learning_rate
        self.high_train_freq = high_train_freq
        self.high_batch_size = high_batch_size
        self.high_tau = high_tau
        self.high_ent_coef = high_ent_coef
        self.high_target_update_interval = high_target_update_interval
        self.high_gradient_steps = high_gradient_steps
        self.high_gamma = high_gamma
        self.high_action_noise = high_action_noise
        self.high_random_exploration = high_random_exploration

        # Agnostic Parameters        
        self.learning_starts = learning_starts
        
        
        # In the original paper, same learning rate is used for all networks
        # self.policy_lr = learning_rate
        # self.qf_lr = learning_rate
        # self.vf_lr = learning_rate
        # Entropy coefficient / Entropy temperature
        # Inverse of the reward scale

        self.graph = None
        self.sess = None
        self.verbose = verbose

        self.full_tensorboard_log = full_tensorboard_log
        self.tensorboard_log = tensorboard_log

        # Low Level Graph Values
        self.low_value_fn = None
        self.low_replay_buffer = None
        self.low_episode_reward = None
        self.low_params = None
        self.low_summary = None
        self.low_policy_tf = None
        self.low_target_entropy = low_target_entropy

        self.low_obs_target = None
        self.low_target_policy = None
        self.low_actions_ph = None
        self.low_rewards_ph = None
        self.low_terminals_ph = None
        self.low_observations_ph = None
        self.low_action_target = None
        self.low_next_observations_ph = None
        self.low_value_target = None
        self.low_step_ops = None
        self.low_target_update_op = None
        self.low_infos_names = None
        self.low_entropy = None
        self.low_target_params = None
        self.low_learning_rate_ph = None
        self.low_processed_obs_ph = None
        self.low_processed_next_obs_ph = None
        self.low_log_ent_coef = None

        # High Level Graph Values
        self.high_value_fn = None
        self.high_replay_buffer = None
        self.high_episode_reward = None
        self.high_params = None
        self.high_summary = None
        self.high_policy_tf = None
        self.high_target_entropy = high_target_entropy

        self.high_obs_target = None
        self.high_target_policy = None
        self.high_actions_ph = None
        self.high_rewards_ph = None
        self.high_terminals_ph = None
        self.high_observations_ph = None
        self.high_action_target = None
        self.high_next_observations_ph = None
        self.high_value_target = None
        self.high_step_ops = None
        self.high_target_update_op = None
        self.high_infos_names = None
        self.high_entropy = None
        self.high_target_params = None
        self.high_learning_rate_ph = None
        self.high_processed_obs_ph = None
        self.high_processed_next_obs_ph = None
        self.high_log_ent_coef = None

        if _init_setup_model:
            self.setup_model()

    
    def _get_pretrain_placeholders(self):
        low_policy = self.low_policy_tf
        high_policy = self.high_policy_tf
        # Rescale
        low_deterministic_action = unscale_action(self.low_action_space, self.low_deterministic_action)
        high_deterministic_action = unscale_action(self.high_action_space, self.high_deterministic_action)

        return (low_policy.obs_ph, self.low_actions_ph, low_deterministic_action), \
                (high_policy.obs_ph, self.high_actions_ph, high_deterministic_action)

    def setup_model(self):
        with SetVerbosity(self.verbose):
            self.graph = tf.Graph()
            with self.graph.as_default():
                self.set_random_seed(self.seed)
                self.sess = tf_util.make_session(num_cpu=self.n_cpu_tf_sess, graph=self.graph)

                self.low_replay_buffer = ReplayBuffer(self.low_buffer_size)
                self.high_replay_buffer = ReplayBuffer(self.high_buffer_size)

                with tf.variable_scope("low", reuse=False):
                    low_summary_vars = list()

                    with tf.variable_scope("input", reuse=False):
                        # Create policy and target TF objects
                        self.low_policy_tf = self.low_policy(self.sess, self.low_observation_space, self.low_action_space,
                                                    **self.low_policy_kwargs)
                        self.low_target_policy = self.low_policy(self.sess, self.low_observation_space, self.low_action_space,
                                                        **self.low_policy_kwargs)

                        # Initialize Placeholders
                        self.low_observations_ph = self.low_policy_tf.obs_ph
                        # Normalized observation for pixels
                        self.low_processed_obs_ph = self.low_policy_tf.processed_obs
                        self.low_next_observations_ph = self.low_target_policy.obs_ph
                        self.low_processed_next_obs_ph = self.low_target_policy.processed_obs
                        self.low_action_target = self.low_target_policy.action_ph
                        self.low_terminals_ph = tf.placeholder(tf.float32, shape=(None, 1), name='terminals')
                        self.low_rewards_ph = tf.placeholder(tf.float32, shape=(None, 1), name='rewards')
                        self.low_actions_ph = tf.placeholder(tf.float32, shape=(None,) + self.low_action_space.shape,
                                                        name='actions')
                        self.low_learning_rate_ph = tf.placeholder(tf.float32, [], name="learning_rate_ph")

                    with tf.variable_scope("model", reuse=False):
                        # Create the policy
                        # first return value corresponds to deterministic actions
                        # policy_out corresponds to stochastic actions, used for training
                        # logp_pi is the log probabilty of actions taken by the policy
                        self.low_deterministic_action, policy_out, logp_pi = self.low_policy_tf.make_actor(self.low_processed_obs_ph)
                        # Monitor the entropy of the policy,
                        # this is not used for training
                        self.low_entropy = tf.reduce_mean(self.low_policy_tf.entropy)
                        #  Use two Q-functions to improve performance by reducing overestimation bias.
                        qf1, qf2, value_fn = self.low_policy_tf.make_critics(self.low_processed_obs_ph, self.low_actions_ph,
                                                                        create_qf=True, create_vf=True)
                        qf1_pi, qf2_pi, _ = self.low_policy_tf.make_critics(self.low_processed_obs_ph,
                                                                        policy_out, create_qf=True, create_vf=False,
                                                                        reuse=True)

                        # Target entropy is used when learning the entropy coefficient
                        if self.low_target_entropy == 'auto':
                            # automatically set target entropy if needed
                            self.low_target_entropy = -np.prod(self.low_action_space.shape).astype(np.float32)
                        else:
                            # Force conversion
                            # this will also throw an error for unexpected string
                            self.low_target_entropy = float(self.low_target_entropy)

                        # The entropy coefficient or entropy can be learned automatically
                        # see Automating Entropy Adjustment for Maximum Entropy RL section
                        # of https://arxiv.org/abs/1812.05905
                        if isinstance(self.low_ent_coef, str) and self.low_ent_coef.startswith('auto'):
                            # Default initial value of ent_coef when learned
                            init_value = 1.0
                            if '_' in self.low_ent_coef:
                                init_value = float(self.low_ent_coef.split('_')[1])
                                assert init_value > 0., "The initial value of ent_coef must be greater than 0"

                            self.low_log_ent_coef = tf.get_variable('log_ent_coef', dtype=tf.float32,
                                                                initializer=np.log(init_value).astype(np.float32))
                            self.low_ent_coef = tf.exp(self.low_log_ent_coef)
                        else:
                            # Force conversion to float
                            # this will throw an error if a malformed string (different from 'auto')
                            # is passed
                            self.low_ent_coef = float(self.low_ent_coef)

                    with tf.variable_scope("target", reuse=False):
                        # Create the value network
                        _, _, value_target = self.low_target_policy.make_critics(self.low_processed_next_obs_ph,
                                                                            create_qf=False, create_vf=True)
                        self.low_value_target = value_target

                    with tf.variable_scope("loss", reuse=False):
                        # Take the min of the two Q-Values (Double-Q Learning)
                        min_qf_pi = tf.minimum(qf1_pi, qf2_pi)

                        # Target for Q value regression
                        q_backup = tf.stop_gradient(
                            self.low_rewards_ph +
                            (1 - self.low_terminals_ph) * self.low_gamma * self.low_value_target
                        )

                        # Compute Q-Function loss
                        # TODO: test with huber loss (it would avoid too high values)
                        qf1_loss = 0.5 * tf.reduce_mean((q_backup - qf1) ** 2)
                        qf2_loss = 0.5 * tf.reduce_mean((q_backup - qf2) ** 2)

                        # Compute the entropy temperature loss
                        # it is used when the entropy coefficient is learned
                        ent_coef_loss, entropy_optimizer = None, None
                        if not isinstance(self.low_ent_coef, float):
                            ent_coef_loss = -tf.reduce_mean(
                                self.low_log_ent_coef * tf.stop_gradient(logp_pi + self.low_target_entropy))
                            entropy_optimizer = tf.train.AdamOptimizer(learning_rate=self.low_learning_rate_ph)

                        # Compute the policy loss
                        # Alternative: policy_kl_loss = tf.reduce_mean(logp_pi - min_qf_pi)
                        policy_kl_loss = tf.reduce_mean(self.low_ent_coef * logp_pi - qf1_pi)

                        # NOTE: in the original implementation, they have an additional
                        # regularization loss for the gaussian parameters
                        # this is not used for now
                        # policy_loss = (policy_kl_loss + policy_regularization_loss)
                        policy_loss = policy_kl_loss


                        # Target for value fn regression
                        # We update the vf towards the min of two Q-functions in order to
                        # reduce overestimation bias from function approximation error.
                        v_backup = tf.stop_gradient(min_qf_pi - self.low_ent_coef * logp_pi)
                        value_loss = 0.5 * tf.reduce_mean((value_fn - v_backup) ** 2)

                        values_losses = qf1_loss + qf2_loss + value_loss

                        # Policy train op
                        # (has to be separate from value train op, because min_qf_pi appears in policy_loss)
                        policy_optimizer = tf.train.AdamOptimizer(learning_rate=self.low_learning_rate_ph)
                        policy_train_op = policy_optimizer.minimize(policy_loss, var_list=get_vars('low/model/pi'))

                        # Value train op
                        value_optimizer = tf.train.AdamOptimizer(learning_rate=self.low_learning_rate_ph)
                        values_params = get_vars('low/model/values_fn')

                        source_params = get_vars("low/model/values_fn/vf")
                        target_params = get_vars("low/target/values_fn/vf")

                        # Polyak averaging for target variables
                        self.low_target_update_op = [
                            tf.assign(target, (1 - self.low_tau) * target + self.low_tau * source)
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

                            self.low_infos_names = ['low_policy_loss', 'low_qf1_loss', 'low_qf2_loss', 'low_value_loss', 'low_entropy']
                            # All ops to call during one training step
                            self.low_step_ops = [policy_loss, qf1_loss, qf2_loss,
                                            value_loss, qf1, qf2, value_fn, logp_pi,
                                            self.low_entropy, policy_train_op, train_values_op]

                            # Add entropy coefficient optimization operation if needed
                            if ent_coef_loss is not None:
                                with tf.control_dependencies([train_values_op]):
                                    ent_coef_op = entropy_optimizer.minimize(ent_coef_loss, var_list=self.low_log_ent_coef)
                                    self.low_infos_names += ['low_ent_coef_loss', 'low_ent_coef']
                                    self.low_step_ops += [ent_coef_op, ent_coef_loss, self.low_ent_coef]

                        # Monitor losses and entropy in tensorboard
                        low_summary_vars.append(tf.summary.scalar('policy_loss', policy_loss))
                        low_summary_vars.append(tf.summary.scalar('qf1_loss', qf1_loss))
                        low_summary_vars.append(tf.summary.scalar('qf2_loss', qf2_loss))
                        low_summary_vars.append(tf.summary.scalar('value_loss', value_loss))
                        low_summary_vars.append(tf.summary.scalar('entropy', self.low_entropy))
                        if ent_coef_loss is not None:
                            low_summary_vars.append(tf.summary.scalar('ent_coef_loss', ent_coef_loss))
                            low_summary_vars.append(tf.summary.scalar('ent_coef', self.low_ent_coef))

                        low_summary_vars.append(tf.summary.scalar('learning_rate', tf.reduce_mean(self.low_learning_rate_ph)))

                    # Retrieve parameters that must be saved
                    self.low_params = get_vars("low/model")
                    self.low_target_params = get_vars("low/target/values_fn/vf")

                    # Initialize Variables and target network
                    with self.sess.as_default():
                        low_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='low')
                        self.sess.run(tf.variables_initializer(low_vars))                        
                        self.sess.run(target_init_op)

                    self.low_summary = tf.summary.merge(low_summary_vars)

                with tf.variable_scope("high", reuse=False):
                    high_summary_vars = list()

                    with tf.variable_scope("input", reuse=False):
                        # Create policy and target TF objects
                        self.high_policy_tf = self.high_policy(self.sess, self.high_observation_space, self.high_action_space,
                                                    **self.high_policy_kwargs)
                        self.high_target_policy = self.high_policy(self.sess, self.high_observation_space, self.high_action_space,
                                                        **self.high_policy_kwargs)

                        # Initialize Placeholders
                        self.high_observations_ph = self.high_policy_tf.obs_ph
                        # Normalized observation for pixels
                        self.high_processed_obs_ph = self.high_policy_tf.processed_obs
                        self.high_next_observations_ph = self.high_target_policy.obs_ph
                        self.high_processed_next_obs_ph = self.high_target_policy.processed_obs
                        self.high_action_target = self.high_target_policy.action_ph
                        self.high_terminals_ph = tf.placeholder(tf.float32, shape=(None, 1), name='terminals')
                        self.high_rewards_ph = tf.placeholder(tf.float32, shape=(None, 1), name='rewards')
                        self.high_actions_ph = tf.placeholder(tf.float32, shape=(None,) + self.high_action_space.shape,
                                                        name='actions')
                        self.high_learning_rate_ph = tf.placeholder(tf.float32, [], name="learning_rate_ph")

                    with tf.variable_scope("model", reuse=False):
                        # Create the policy
                        # first return value corresponds to deterministic actions
                        # policy_out corresponds to stochastic actions, used for training
                        # logp_pi is the log probabilty of actions taken by the policy
                        self.high_deterministic_action, policy_out, logp_pi = self.high_policy_tf.make_actor(self.high_processed_obs_ph)
                        # Monitor the entropy of the policy,
                        # this is not used for training
                        self.high_entropy = tf.reduce_mean(self.high_policy_tf.entropy)
                        #  Use two Q-functions to improve performance by reducing overestimation bias.
                        qf1, qf2, value_fn = self.high_policy_tf.make_critics(self.high_processed_obs_ph, self.high_actions_ph,
                                                                        create_qf=True, create_vf=True)
                        qf1_pi, qf2_pi, _ = self.high_policy_tf.make_critics(self.high_processed_obs_ph,
                                                                        policy_out, create_qf=True, create_vf=False,
                                                                        reuse=True)

                        # Target entropy is used when learning the entropy coefficient
                        if self.high_target_entropy == 'auto':
                            # automatically set target entropy if needed
                            self.high_target_entropy = -np.prod(self.high_action_space.shape).astype(np.float32)
                        else:
                            # Force conversion
                            # this will also throw an error for unexpected string
                            self.high_target_entropy = float(self.high_target_entropy)

                        # The entropy coefficient or entropy can be learned automatically
                        # see Automating Entropy Adjustment for Maximum Entropy RL section
                        # of https://arxiv.org/abs/1812.05905
                        if isinstance(self.high_ent_coef, str) and self.high_ent_coef.startswith('auto'):
                            # Default initial value of ent_coef when learned
                            init_value = 1.0
                            if '_' in self.high_ent_coef:
                                init_value = float(self.high_ent_coef.split('_')[1])
                                assert init_value > 0., "The initial value of ent_coef must be greater than 0"

                            self.high_log_ent_coef = tf.get_variable('log_ent_coef', dtype=tf.float32,
                                                                initializer=np.log(init_value).astype(np.float32))
                            self.high_ent_coef = tf.exp(self.high_log_ent_coef)
                        else:
                            # Force conversion to float
                            # this will throw an error if a malformed string (different from 'auto')
                            # is passed
                            self.high_ent_coef = float(self.high_ent_coef)

                    with tf.variable_scope("target", reuse=False):
                        # Create the value network
                        _, _, value_target = self.high_target_policy.make_critics(self.high_processed_next_obs_ph,
                                                                            create_qf=False, create_vf=True)
                        self.high_value_target = value_target

                    with tf.variable_scope("loss", reuse=False):
                        # Take the min of the two Q-Values (Double-Q Learning)
                        min_qf_pi = tf.minimum(qf1_pi, qf2_pi)

                        # Target for Q value regression
                        q_backup = tf.stop_gradient(
                            self.high_rewards_ph +
                            (1 - self.high_terminals_ph) * self.high_gamma * self.high_value_target
                        )

                        # Compute Q-Function loss
                        # TODO: test with huber loss (it would avoid too high values)
                        qf1_loss = 0.5 * tf.reduce_mean((q_backup - qf1) ** 2)
                        qf2_loss = 0.5 * tf.reduce_mean((q_backup - qf2) ** 2)

                        # Compute the entropy temperature loss
                        # it is used when the entropy coefficient is learned
                        ent_coef_loss, entropy_optimizer = None, None
                        if not isinstance(self.high_ent_coef, float):
                            ent_coef_loss = -tf.reduce_mean(
                                self.high_log_ent_coef * tf.stop_gradient(logp_pi + self.high_target_entropy))
                            entropy_optimizer = tf.train.AdamOptimizer(learning_rate=self.high_learning_rate_ph)

                        # Compute the policy loss
                        # Alternative: policy_kl_loss = tf.reduce_mean(logp_pi - min_qf_pi)
                        policy_kl_loss = tf.reduce_mean(self.high_ent_coef * logp_pi - qf1_pi)

                        # NOTE: in the original implementation, they have an additional
                        # regularization loss for the gaussian parameters
                        # this is not used for now
                        # policy_loss = (policy_kl_loss + policy_regularization_loss)
                        policy_loss = policy_kl_loss


                        # Target for value fn regression
                        # We update the vf towards the min of two Q-functions in order to
                        # reduce overestimation bias from function approximation error.
                        v_backup = tf.stop_gradient(min_qf_pi - self.high_ent_coef * logp_pi)
                        value_loss = 0.5 * tf.reduce_mean((value_fn - v_backup) ** 2)

                        values_losses = qf1_loss + qf2_loss + value_loss

                        # Policy train op
                        # (has to be separate from value train op, because min_qf_pi appears in policy_loss)
                        policy_optimizer = tf.train.AdamOptimizer(learning_rate=self.high_learning_rate_ph)
                        policy_train_op = policy_optimizer.minimize(policy_loss, var_list=get_vars('high/model/pi'))

                        # Value train op
                        value_optimizer = tf.train.AdamOptimizer(learning_rate=self.high_learning_rate_ph)
                        values_params = get_vars('high/model/values_fn')

                        source_params = get_vars("high/model/values_fn/vf")
                        target_params = get_vars("high/target/values_fn/vf")

                        # Polyak averaging for target variables
                        self.high_target_update_op = [
                            tf.assign(target, (1 - self.high_tau) * target + self.high_tau * source)
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

                            self.high_infos_names = ['high_policy_loss', 'high_qf1_loss', 'high_qf2_loss', 'high_value_loss', 'high_entropy']
                            # All ops to call during one training step
                            self.high_step_ops = [policy_loss, qf1_loss, qf2_loss,
                                            value_loss, qf1, qf2, value_fn, logp_pi,
                                            self.high_entropy, policy_train_op, train_values_op]

                            # Add entropy coefficient optimization operation if needed
                            if ent_coef_loss is not None:
                                with tf.control_dependencies([train_values_op]):
                                    ent_coef_op = entropy_optimizer.minimize(ent_coef_loss, var_list=self.high_log_ent_coef)
                                    self.high_infos_names += ['high_ent_coef_loss', 'high_ent_coef']
                                    self.high_step_ops += [ent_coef_op, ent_coef_loss, self.high_ent_coef]

                        # Monitor losses and entropy in tensorboard
                        high_summary_vars.append(tf.summary.scalar('policy_loss', policy_loss))
                        high_summary_vars.append(tf.summary.scalar('qf1_loss', qf1_loss))
                        high_summary_vars.append(tf.summary.scalar('qf2_loss', qf2_loss))
                        high_summary_vars.append(tf.summary.scalar('value_loss', value_loss))
                        high_summary_vars.append(tf.summary.scalar('entropy', self.high_entropy))
                        if ent_coef_loss is not None:
                            high_summary_vars.append(tf.summary.scalar('ent_coef_loss', ent_coef_loss))
                            high_summary_vars.append(tf.summary.scalar('ent_coef', self.high_ent_coef))

                        high_summary_vars.append(tf.summary.scalar('learning_rate', tf.reduce_mean(self.high_learning_rate_ph)))

                    # Retrieve parameters that must be saved
                    self.high_params = get_vars("high/model")
                    self.high_target_params = get_vars("high/target/values_fn/vf")

                    # Initialize Variables and target network
                    with self.sess.as_default():
                        high_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='high')
                        self.sess.run(tf.variables_initializer(high_vars))                        
                        self.sess.run(target_init_op)

                    self.high_summary = tf.summary.merge(high_summary_vars)

                
    def _low_train_step(self, step, writer, learning_rate):
        # Sample a batch from the replay buffer
        batch = self.low_replay_buffer.sample(self.low_batch_size)
        batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones = batch

        feed_dict = {
            self.low_observations_ph: batch_obs,
            self.low_actions_ph: batch_actions,
            self.low_next_observations_ph: batch_next_obs,
            self.low_rewards_ph: batch_rewards.reshape(self.low_batch_size, -1),
            self.low_terminals_ph: batch_dones.reshape(self.low_batch_size, -1),
            self.low_learning_rate_ph: learning_rate
        }

        # out  = [policy_loss, qf1_loss, qf2_loss,
        #         value_loss, qf1, qf2, value_fn, logp_pi,
        #         self.entropy, policy_train_op, train_values_op]

        # Do one gradient step
        # and optionally compute log for tensorboard
        if writer is not None:
            out = self.sess.run([self.low_summary] + self.low_step_ops, feed_dict)
            summary = out.pop(0)
            writer.add_summary(summary, step)
        else:
            out = self.sess.run(self.low_step_ops, feed_dict)

        # Unpack to monitor losses and entropy
        policy_loss, qf1_loss, qf2_loss, value_loss, *values = out
        # qf1, qf2, value_fn, logp_pi, entropy, *_ = values
        entropy = values[4]

        if self.low_log_ent_coef is not None:
            ent_coef_loss, ent_coef = values[-2:]
            return policy_loss, qf1_loss, qf2_loss, value_loss, entropy, ent_coef_loss, ent_coef

        return policy_loss, qf1_loss, qf2_loss, value_loss, entropy

    def _high_train_step(self, step, writer, learning_rate):
        # Sample a batch from the replay buffer
        batch = self.high_replay_buffer.sample(self.high_batch_size)
        batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones = batch

        feed_dict = {
            self.high_observations_ph: batch_obs,
            self.high_actions_ph: batch_actions,
            self.high_next_observations_ph: batch_next_obs,
            self.high_rewards_ph: batch_rewards.reshape(self.high_batch_size, -1),
            self.high_terminals_ph: batch_dones.reshape(self.high_batch_size, -1),
            self.high_learning_rate_ph: learning_rate
        }

        # out  = [policy_loss, qf1_loss, qf2_loss,
        #         value_loss, qf1, qf2, value_fn, logp_pi,
        #         self.entropy, policy_train_op, train_values_op]

        # Do one gradient step
        # and optionally compute log for tensorboard
        if writer is not None:
            out = self.sess.run([self.high_summary] + self.high_step_ops, feed_dict)
            summary = out.pop(0)
            writer.add_summary(summary, step)
        else:
            out = self.sess.run(self.high_step_ops, feed_dict)

        # Unpack to monitor losses and entropy
        policy_loss, qf1_loss, qf2_loss, value_loss, *values = out
        # qf1, qf2, value_fn, logp_pi, entropy, *_ = values
        entropy = values[4]

        if self.high_log_ent_coef is not None:
            ent_coef_loss, ent_coef = values[-2:]
            return policy_loss, qf1_loss, qf2_loss, value_loss, entropy, ent_coef_loss, ent_coef

        return policy_loss, qf1_loss, qf2_loss, value_loss, entropy

    def learn(self, total_timesteps, callback=None,
              log_interval=4, tb_log_name="SAC", reset_num_timesteps=True, replay_wrapper=None,
              high_training_starts=0):

        new_tb_log = self._init_num_timesteps(reset_num_timesteps)

        if replay_wrapper is not None:
            self.low_replay_buffer = replay_wrapper(self.low_replay_buffer)
            self.high_replay_buffer = replay_wrapper(self.high_replay_buffer)

        with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name, new_tb_log) \
                as writer:

            self._setup_learn()

            # Transform to callable if needed
            self.low_learning_rate = get_schedule_fn(self.low_learning_rate)
            self.high_learning_rate = get_schedule_fn(self.high_learning_rate)

            # Initial learning rate
            low_current_lr = self.low_learning_rate(1)
            high_current_lr = self.high_learning_rate(1)

            start_time = time.time()
            episode_rewards = [0.0]
            episode_successes = []
            if self.low_action_noise is not None:
                self.low_action_noise.reset()
            if self.high_action_noise is not None:
                self.high_action_noise.reset()

            obs = self.env.reset()
            self.high_episode_reward = np.zeros((1,))
            self.low_episode_reward = np.zeros((1,))
            
            high_ep_info_buf = deque(maxlen=100)
            low_ep_info_buf = deque(maxlen=100)

            low_n_updates = 0
            high_n_updates = 0
            low_infos_values = []
            high_infos_values = []

            low_step = 0

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
                if self.num_timesteps < high_training_starts or self.num_timesteps < self.learning_starts or np.random.rand() < self.high_random_exploration:
                    # actions sampled from action space are from range specific to the environment
                    # but algorithm operates on tanh-squashed actions therefore simple scaling is used
                    high_unscaled_action = self.high_action_space.sample()
                    high_action = scale_action(self.high_action_space, high_unscaled_action)
                    if self.num_timesteps < high_training_starts:
                        self.env.reset()
                else:
                    high_action = self.high_policy_tf.step(obs[None], deterministic=False).flatten()
                    # Add noise to the action (improve exploration,
                    # not needed in general)
                    if self.high_action_noise is not None:
                        high_action = np.clip(high_action + self.high_action_noise(), -1, 1)
                    # inferred actions need to be transformed to environment action_space before stepping
                    high_unscaled_action = unscale_action(self.high_action_space, high_action)

                def low_predict_func(low_obs):
                    if self.num_timesteps < self.learning_starts or np.random.rand() < self.low_random_exploration:
                        low_unscaled_action = self.low_action_space.sample()
                        low_action = scale_action(self.low_action_space, low_unscaled_action)
                    else:
                        low_action = self.low_policy_tf.step(low_obs[None], deterministic=False).flatten()
                        if self.low_action_noise is not None:
                            low_action = np.clip(high_action + self.low_action_noise(), -1, 1)
                        low_unscaled_action = unscale_action(self.low_action_space, low_action)
                    return low_action, low_unscaled_action

                new_obs, reward, done, info = self.env.step((high_unscaled_action, low_predict_func))

                # Store Low Transitions in the replay buffer:
                low_obs = info['obs']
                low_actions = info['actions']
                low_rewards = info['rewards']
                low_dones = info['dones']
                low_ep_info_buf.append(info['low_ep_info'])
                
                for i in range(len(low_actions)):
                    self.low_replay_buffer.add(low_obs[i], low_actions[i], low_rewards[i], low_obs[i+1], float(low_dones[i]))
                    # Run Low Level Policy Updates                    
                    # Train Low Level
                    if low_step % self.low_train_freq == 0:
                        mb_infos_vals = []
                        # Update policy, critics and target networks
                        for grad_step in range(self.low_gradient_steps):
                            # Break if the warmup phase is not over
                            # or if there are not enough samples in the replay buffer
                            if not self.low_replay_buffer.can_sample(self.low_batch_size) \
                            or self.num_timesteps < self.learning_starts:
                                break
                            low_n_updates += 1
                            # Compute current learning_rate
                            frac = 1.0 - step / total_timesteps
                            current_lr = self.low_learning_rate(frac)
                            # Update policy and critics (q functions)
                            mb_infos_vals.append(self._low_train_step(low_step, writer, current_lr))
                            # Update target network
                            if (low_step + grad_step) % self.low_target_update_interval == 0:
                                # Update target network
                                self.sess.run(self.low_target_update_op)
                        # Log losses and entropy, useful for monitor training
                        if len(mb_infos_vals) > 0:
                            low_infos_values = np.mean(mb_infos_vals, axis=0)

                    low_step += 1

                # Store high transition in the replay buffer.
                self.high_replay_buffer.add(obs, high_action, reward, new_obs, float(done))
                obs = new_obs

                # Retrieve high reward and episode length if using Monitor wrapper
                maybe_ep_info = info.get('episode')
                if maybe_ep_info is not None:
                    high_ep_info_buf.extend([maybe_ep_info])

                if writer is not None:
                    # Write reward per episode to tensorboard
                    # TODO: add low level
                    ep_reward = np.array([reward]).reshape((1, -1))
                    ep_done = np.array([done]).reshape((1, -1))
                    self.high_episode_reward = total_episode_reward_logger(self.high_episode_reward, ep_reward,
                                                                      ep_done, writer, self.num_timesteps)

                # Train High Level
                if step % self.high_train_freq == 0:
                    mb_infos_vals = []
                    # Update policy, critics and target networks
                    for grad_step in range(self.high_gradient_steps):
                        # Break if the warmup phase is not over
                        # or if there are not enough samples in the replay buffer
                        if not self.high_replay_buffer.can_sample(self.high_batch_size) \
                           or self.num_timesteps < self.learning_starts:
                            break
                        high_n_updates += 1
                        # Compute current learning_rate
                        frac = 1.0 - step / total_timesteps
                        current_lr = self.high_learning_rate(frac)
                        # Update policy and critics (q functions)
                        mb_infos_vals.append(self._high_train_step(step, writer, current_lr))
                        # Update target network
                        if (step + grad_step) % self.high_target_update_interval == 0:
                            # Update target network
                            self.sess.run(self.high_target_update_op)
                    # Log losses and entropy, useful for monitor training
                    if len(mb_infos_vals) > 0:
                        high_infos_values = np.mean(mb_infos_vals, axis=0)

                # High Level Book Keeping for rewards / dones
                episode_rewards[-1] += reward
                if done:
                    if self.high_action_noise is not None:
                        self.high_action_noise.reset()
                    if not isinstance(self.env, VecEnv):
                        obs = self.env.reset()
                    episode_rewards.append(0.0)

                    maybe_is_success = info.get('is_success')
                    if maybe_is_success is not None:
                        episode_successes.append(float(maybe_is_success))

                if len(episode_rewards[-101:-1]) == 0:
                    mean_reward = -np.inf
                else:
                    mean_reward = round(float(np.mean(episode_rewards[-101:-1])), 1)

                num_episodes = len(episode_rewards)
                self.num_timesteps += 1
                # Display training infos
                if self.verbose >= 1 and done and log_interval is not None and len(episode_rewards) % log_interval == 0:
                    fps = int(step / (time.time() - start_time))
                    logger.logkv("episodes", num_episodes)
                    logger.logkv("mean 100 episode reward", mean_reward)
                    if len(low_ep_info_buf) > 0 and len(low_ep_info_buf[0]) > 0:
                        logger.logkv('low_ep_rewmean', safe_mean([ep_info['r'] for ep_info in low_ep_info_buf]))
                        logger.logkv('low_eplenmean', safe_mean([ep_info['l'] for ep_info in low_ep_info_buf]))
                    if len(high_ep_info_buf) > 0 and len(high_ep_info_buf[0]) > 0:
                        logger.logkv('high_ep_rewmean', safe_mean([ep_info['r'] for ep_info in high_ep_info_buf]))
                        logger.logkv('high_eplenmean', safe_mean([ep_info['l'] for ep_info in high_ep_info_buf]))
                    logger.logkv("low_n_updates", low_n_updates)
                    logger.logkv("high_n_updates", high_n_updates)
                    logger.logkv("low_current_lr", low_current_lr)
                    logger.logkv("high_current_lr", high_current_lr)
                    logger.logkv("fps", fps)
                    logger.logkv('time_elapsed', int(time.time() - start_time))
                    if len(episode_successes) > 0:
                        logger.logkv("success rate", np.mean(episode_successes[-100:]))
                    if len(low_infos_values) > 0:
                        for (name, val) in zip(self.low_infos_names, low_infos_values):
                            logger.logkv(name, val)
                    if len(high_infos_values) > 0:
                        for (name, val) in zip(self.high_infos_names, high_infos_values):
                            logger.logkv(name, val)
                    logger.logkv("total timesteps", self.num_timesteps)
                    logger.dumpkvs()
                    # Reset infos:
                    low_infos_values = []
                    high_infos_values = []
            return self

    def action_probability(self, observation, state=None, mask=None, actions=None, logp=False):
        if actions is not None:
            raise ValueError("Error: SAC does not have action probabilities.")

        warnings.warn("Even though SAC has a Gaussian policy, it cannot return a distribution as it "
                      "is squashed by a tanh before being scaled and ouputed.")

        return None

    def skill_probability(self, observation, state=None, mask=None, actions=None, logp=False):
        if actions is not None:
            raise ValueError("Error: SAC does not have skill probabilities.")

        warnings.warn("Even though SAC has a Gaussian policy, it cannot return a distribution as it "
                      "is squashed by a tanh before being scaled and ouputed.")

        return None

    def predict(self, observation, state=None, mask=None, deterministic=True):
        observation = np.array(observation)
        vectorized_env = self._is_vectorized_observation(observation, self.low_observation_space)

        observation = observation.reshape((-1,) + self.low_observation_space.shape)
        actions = self.low_policy_tf.step(observation, deterministic=deterministic)
        actions = actions.reshape((-1,) + self.low_action_space.shape)  # reshape to the correct action shape
        actions = unscale_action(self.low_action_space, actions) # scale the output for the prediction

        if not vectorized_env:
            actions = actions[0]

        return actions, None

    def predict_skill(self, observation, state=None, mask=None, deterministic=True):
        observation = np.array(observation)
        vectorized_env = self._is_vectorized_observation(observation, self.high_observation_space)

        observation = observation.reshape((-1,) + self.high_observation_space.shape)
        actions = self.high_policy_tf.step(observation, deterministic=deterministic)
        actions = actions.reshape((-1,) + self.high_action_space.shape)  # reshape to the correct action shape
        actions = unscale_action(self.high_action_space, actions) # scale the output for the prediction

        if not vectorized_env:
            actions = actions[0]

        return actions, None

    def get_parameter_list(self):
        return (self.low_params + self.low_target_params), (self.high_params +self.high_target_params)

    def save(self, low_save_path, high_save_path, cloudpickle=False):
        low_data = {
            "learning_rate": self.low_learning_rate,
            "buffer_size": self.low_buffer_size,
            "learning_starts": self.learning_starts,
            "train_freq": self.low_train_freq,
            "batch_size": self.low_batch_size,
            "tau": self.low_tau,
            "ent_coef": self.low_ent_coef if isinstance(self.low_ent_coef, float) else 'auto',
            "target_entropy": self.low_target_entropy,
            # Should we also store the replay buffer?
            # this may lead to high memory usage
            # with all transition inside
            # "replay_buffer": self.replay_buffer
            "gamma": self.low_gamma,
            "verbose": self.verbose,
            "observation_space": self.low_observation_space,
            "action_space": self.low_action_space,
            "policy": self.low_policy,
            "n_envs": self.n_envs,
            "n_cpu_tf_sess": self.n_cpu_tf_sess,
            "seed": self.seed,
            "action_noise": self.low_action_noise,
            "random_exploration": self.low_random_exploration,
            "_vectorize_action": self._vectorize_action,
            "policy_kwargs": self.low_policy_kwargs
        }

        high_data = {
            "learning_rate": self.high_learning_rate,
            "buffer_size": self.high_buffer_size,
            "learning_starts": self.learning_starts,
            "train_freq": self.high_train_freq,
            "batch_size": self.high_batch_size,
            "tau": self.high_tau,
            "ent_coef": self.high_ent_coef if isinstance(self.high_ent_coef, float) else 'auto',
            "target_entropy": self.high_target_entropy,
            # Should we also store the replay buffer?
            # this may lead to high memory usage
            # with all transition inside
            # "replay_buffer": self.replay_buffer
            "gamma": self.high_gamma,
            "verbose": self.verbose,
            "observation_space": self.high_observation_space,
            "action_space": self.high_action_space,
            "policy": self.high_policy,
            "n_envs": self.n_envs,
            "n_cpu_tf_sess": self.n_cpu_tf_sess,
            "seed": self.seed,
            "action_noise": self.high_action_noise,
            "random_exploration": self.high_random_exploration,
            "_vectorize_action": self._vectorize_action,
            "policy_kwargs": self.high_policy_kwargs
        }

        low_params_to_save, high_params_to_save = self.get_parameters(trim_prefix=True)

        self._save_to_file(low_save_path, data=low_data, params=low_params_to_save, cloudpickle=cloudpickle)
        self._save_to_file(high_save_path, data=high_data, params=high_params_to_save, cloudpickle=cloudpickle)

if __name__ == "__main__":
    from stable_baselines.sac.policies import MlpPolicy
    from bot_transfer.envs.hierarchical import JointOPEnv
    from bot_transfer.envs.point_mass import PointMassSmallMJ
    from gym.wrappers import TimeLimit
    from stable_baselines.common.vec_env import DummyVecEnv
    from stable_baselines.bench import Monitor

    def make_env():
        env = JointOPEnv(PointMassSmallMJ(k=20))
        env = TimeLimit(env, 50)
        env = Monitor(env, None)
        return env

    env = make_env()
    # env = DummyVecEnv([lambda: env])

    model = HSAC(MlpPolicy, MlpPolicy, env, tensorboard_log="test_hsac", seed=1, verbose=1)
    model.learn(total_timesteps=1000, log_interval=5)
    model.save("test")

    # loaded_test = HPPO1.load("test")
