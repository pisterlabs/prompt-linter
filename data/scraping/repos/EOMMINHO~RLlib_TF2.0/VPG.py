import tensorflow as tf
import numpy as np
import gym
import Utilities
import time
import json
import pdb
import multiprocessing as mp
from Lib.Utils.Logger.Logger import EpochLogger


class VPG_buffer(object):
    """
    A buffer for storing trajectories experienced by a VPG agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    IMPLEMENTATION from OpenAI Spinningup -> Changed to list
    """

    def __init__(self, size, gamma=0.99, lam=0.97):
        self.obs_buf = []
        self.act_buf = []
        self.rew_buf = []
        self.ret_buf = []
        self.logp_buf = []
        self.val_buf = []
        self.adv_buf = []
        self.episodic_reward = []
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size
        self.total_ep = 0

    def store(self, obs, act, rew, logp, val):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        self.obs_buf.append(obs)
        self.act_buf.append(act)
        self.rew_buf.append(rew)
        self.logp_buf.append(logp)
        self.val_buf.append(val)
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        self.episodic_reward.append(sum(self.rew_buf[path_slice]))
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = Utilities.discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = Utilities.discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr
        self.total_ep += 1

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size  # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # change the buffers to tensors
        self.obs_buf = tf.convert_to_tensor(self.obs_buf)
        self.act_buf = tf.convert_to_tensor(self.act_buf)
        self.ret_buf = tf.convert_to_tensor(self.ret_buf)
        self.logp_buf = tf.convert_to_tensor(self.logp_buf)
        self.val_buf = tf.convert_to_tensor(self.val_buf)
        self.adv_buf = tf.convert_to_tensor(self.adv_buf)
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = tf.math.reduce_mean(self.adv_buf), tf.math.reduce_std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        return [self.obs_buf, self.act_buf, self.adv_buf,
                self.ret_buf, self.logp_buf, self.rew_buf, self.total_ep]


class VPG(object):
    gamma: float

    def __init__(self, env_name, logdir, filename, train_time_step=5000, render_step=1000, gamma=0.99, lamb=0.97,
                 v_iters=80, epochs=50,
                 pi_lr=3e-4, vf_lr=1e-3, processes=1, max_per_episode=1000, env_seed=None):
        self.gamma = gamma
        self.lamb = lamb
        self.v_iter = v_iters
        self.epochs = epochs
        self.processes = processes

        self.env = gym.make(env_name)
        self.total_train_time_step = train_time_step
        self.train_time_step = int(self.total_train_time_step * (1. / processes))
        self.render_step = render_step
        self.value_function = None
        self.policy_function = None

        self.policy_optimizer = tf.keras.optimizers.Adam(learning_rate=pi_lr)
        self.value_optimizer = tf.keras.optimizers.Adam(learning_rate=vf_lr)

        self.env_seed = None

        self.epoch_logger = EpochLogger(logdir, filename)

        self._extra_pbuffer = 0

    def set_value_function(self, function):
        self.value_function = function

    def set_policy_function(self, function):
        self.policy_function = function

    @tf.function
    def train_value(self, statebuffer, returnbuffer):
        # fit value function to the return buffer
        with tf.GradientTape() as tape:
            # Update value via gradient descent
            value_buf = self.value_function(statebuffer)
            mse = tf.square(tf.reshape(value_buf, [-1]) - returnbuffer) * (1 / len(returnbuffer))
            value_loss = tf.reduce_sum(mse)
        value_gradient = tape.gradient(value_loss, self.value_function.trainable_variables)
        self.value_optimizer.apply_gradients(zip(value_gradient, self.value_function.trainable_variables))

    def train_step(self, buf):
        # For k steps do the following
        # collect set of trajectories
        start = time.time()
        with tf.GradientTape(persistent=True) as tape:
            # seed to the environment
            if self.env_seed is not None:
                self.env.seed(seed=self.env_seed)
                self.env_seed += 1000

            obs = self.env.reset()
            for i in range(self.train_time_step):
                # with policy function sample an action
                action_oh, action, log_likelihood = Utilities.obs2action(self.policy_function, obs)
                # for the given s, a, output s', r
                next_obs, reward, done, _ = self.env.step(action.numpy())
                val = Utilities.obs2value(self.value_function, obs)
                # record the state and action
                buf.store(obs, action_oh, reward, log_likelihood, val)
                obs = next_obs

                # if the episode finished or must be truncated
                if done or (i == self.train_time_step - 1):
                    # if it is truncated due to maximum train step
                    if i == self.train_time_step - 1:
                        print("caution: train has truncated on length")
                        # bootstrap the reward to value function approximation
                        boot_val = Utilities.obs2value(self.value_function, obs)
                        buf.finish_path(last_val=boot_val)
                    # if it is not truncated but end of episode (died during episode)
                    else:
                        obs = self.env.reset()
                        # append reward 0 to the final state
                        buf.finish_path(last_val=0)

            # compute loss
            obs_buf, _, adv_buf, ret_buf, logp_buf, rew_buf, num_ep = buf.get()
            policy_loss = -tf.reduce_sum(tf.math.multiply(logp_buf, adv_buf)) * (1. / 5000)

        # Update policy via gradient ascent
        policy_gradient = tape.gradient(policy_loss, self.policy_function.trainable_variables)
        self.policy_optimizer.apply_gradients(zip(policy_gradient, self.policy_function.trainable_variables))
        del tape

        # fit the value function to mean-squared error
        for i in range(self.v_iter):
            self.train_value(obs_buf, ret_buf)

        stat = dict()
        # statistics of return
        ep_ret = np.array(buf.episodic_reward)
        stat["avg_ret"] = np.mean(ep_ret)
        stat["std_ret"] = np.std(ep_ret)
        stat["max_ret"] = np.amax(ep_ret)
        stat["min_ret"] = np.amin(ep_ret)
        # Entropy and KL divergence
        stat["entropy"] = tf.reduce_mean(-logp_buf).numpy()
        stat["KL"] = tf.reduce_mean(self._extra_pbuffer - logp_buf).numpy()
        stat["time"] = time.time() - start

        self.epoch_logger.append_dict(stat)

        self._extra_pbuffer = logp_buf
        return stat

    def train(self):
        # train for epochs
        for epoch in range(self.epochs):
            buf = VPG_buffer(self.total_train_time_step)
            self.train_step(buf)
        # save log to the file
        self.epoch_logger.log_json()

    def render(self):
        """
        Render the agents moves to see how it works
        :return: None
        """
        observation = self.env.reset()
        for i in range(self.render_step):
            self.env.render()

            observation = tf.reshape(observation, [1, observation.shape[0]])
            action_logit = self.policy_function(observation)
            action_logit = tf.squeeze(action_logit)
            action = tf.random.categorical(tf.reshape(action_logit, [1, -1]), 1)[0][0]  # sampling from distribution

            observation, reward, done, info = self.env.step(action.numpy())

            if done:
                break
                # observation = self.env.reset()
        self.env.close()


class value_function(tf.keras.Model):
    """
    Return the value of that state for the given observation
    """

    def __init__(self, input_shape, seed):
        super(value_function, self).__init__()
        self.dense1 = tf.keras.layers.Dense(64, input_shape=input_shape)
        self.dense2 = tf.keras.layers.Dense(64)
        self.dense3 = tf.keras.layers.Dense(1)

        self.relu = tf.keras.layers.ReLU()
        self.tanh = tf.math.tanh

    @tf.function
    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.relu(x)
        x = self.dense2(x)
        x = self.relu(x)
        x = self.dense3(x)

        return x


class policy_function(tf.keras.Model):
    """
    Return the probability of action for the given observation
    """

    def __init__(self, input_shape, output_dim, seed):
        super(policy_function, self).__init__()
        self.dense1 = tf.keras.layers.Dense(64, input_shape=input_shape)
        self.dense2 = tf.keras.layers.Dense(64)
        self.dense3 = tf.keras.layers.Dense(output_dim)

        self.relu = tf.keras.layers.ReLU()
        self.tanh = tf.math.tanh
        # self.softmax = tf.keras.layers.Softmax()

    @tf.function
    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.relu(x)
        x = self.dense2(x)
        x = self.relu(x)
        x = self.dense3(x)

        return x


if __name__ == "__main__":
    seed = 0
    vf = value_function([4, ], seed)
    pf = policy_function([4, ], 2, seed)
    agent = VPG("CartPole-v0", train_time_step=5000, epochs=10, logdir="./", filename="first.txt")
    agent.set_value_function(vf)
    agent.set_policy_function(pf)

    # agent.render()
    agent.train()
