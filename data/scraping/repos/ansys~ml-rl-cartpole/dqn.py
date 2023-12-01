import os
import pickle

import numpy as np


class Diagnostics:
    def __init__(self, active=False):
        self.active = active

    def record_q_swap(self):
        if self.active:
            print("Q-swap")


class StateFormatter:
    """
    This is meant to convert states coming from openai gym into a learner state encoding
    In the simplest of examples, the formatter is an identity function,
    but it may choose to make subtle modifications to the state
    """

    def __init__(self, s_shape_gym):
        self.s_shape = s_shape_gym

    def convert(self, step, s, is_terminal=False):
        return np.r_[s] if not is_terminal else np.full((self.s_shape,), fill_value=np.nan)

    def is_terminal(self, s):
        return np.any(np.isnan(s))


class StateFormatterIndexed:
    """
    This is meant to convert states coming from openai gym into a learner state encoding
    In the simplest of examples, the formatter is an identity function,
    but it may choose to make subtle modifications to the state
    """

    def __init__(self, s_shape_gym):
        self.s_shape = s_shape_gym + 1

    def convert(self, step, s, is_terminal=False):
        return np.r_[step, s] if not is_terminal else np.full((self.s_shape,), fill_value=np.nan)

    def is_terminal(self, s):
        return np.any(np.isnan(s))


class ReplayBuffer:
    def __init__(self, env, hwm0, capacity):
        self.env = env
        self.s_formatter = StateFormatter(self.env.reset().shape[0])
        self.cursor = 0
        self.hwm = 0
        self.b = self.generate_samples(hwm0, capacity)

    def store(self, sample):
        self.b[self.cursor] = sample
        self.cursor = (self.cursor + 1) % self.b.shape[0]
        self.hwm = np.clip(self.hwm + 1, 0, self.b.shape[0])

    def generate_samples(self, hwm0, capacity):
        i_sample = 0
        seq_num = 1
        s = self.s_formatter.convert(seq_num, self.env.reset())
        b = np.full((capacity, self.s_formatter.s_shape * 2 + 2), fill_value=np.nan, dtype=float)
        while i_sample < hwm0:
            a = self.env.action_space.sample()
            sp_gym, r, done, _ = self.env.step(a)
            sp = self.s_formatter.convert(seq_num + 1, sp_gym, done)
            b[i_sample] = np.r_[s, a, r, sp]
            seq_num = 1 if done else seq_num + 1
            s = self.s_formatter.convert(seq_num, self.env.reset()) if done else sp
            i_sample += 1
        self.cursor = self.hwm = hwm0
        return b

    def mini_batch(self, n):
        return self.b[np.random.randint(self.hwm, size=n)]

    def save(self, path, base_name):
        if not os.path.exists(path):
            os.makedirs(path)

        with open(os.path.join(path, base_name + "_rpl.pkl"), "wb") as f:
            pickle.dump(self.s_formatter, f)
            pickle.dump(self.cursor, f)
            pickle.dump(self.hwm, f)

        np.save(os.path.join(path, base_name + "_rpl.npy"), self.b)

    def load(self, path, base_name):
        self.b = np.load(os.path.join(path, base_name + "_rpl.npy"))

        with open(os.path.join(path, base_name + "_dqn.pkl"), "rb") as f:
            self.s_formatter = pickle.load(f)
            self.cursor = pickle.load(f)
            self.hwm = pickle.load(f)


class ClassicDQNLearner:
    def __init__(
        self,
        env_db,
        q_factory,
        layers,
        epsilon,
        gamma,
        n_mini_batch,
        replay_db_warmup,
        replay_db_capacity,
        c_cycle,
        polyak_rate,
        shaper_fn=None,
        diagnostics=False,
        double_dqn=True,
        output_path=None,
        output_name=None,
    ):
        self.a_shape = 1
        self.n_actions = env_db.action_space.n
        self.s_formatter = StateFormatter(env_db.reset().shape[0])
        self.n_mini_batch = n_mini_batch
        self.epsilon = epsilon
        self.gamma = gamma
        self.t_cycle = c_cycle
        self.shaper_fn = shaper_fn
        self.polyak_rate = polyak_rate
        self.output_path = output_path
        self.output_name = output_name

        self.s = None
        self.a = None
        self.t = 0
        self.seq_num = 1
        self.diagnostics = Diagnostics(diagnostics)

        # initialize reply memory D with capacity N
        self.buffer = ReplayBuffer(env_db, replay_db_warmup, replay_db_capacity)

        # initialize action-value function Q with random weights theta
        self.q = q_factory(layers, self.s_formatter, self.a_shape, self.n_actions, self.n_mini_batch)
        self.q_target = q_factory(layers, self.s_formatter, self.a_shape, self.n_actions, self.n_mini_batch)
        self.q_double = self.q if double_dqn else self.q_target

        # initialize action-value function Q_hat with weights theta_minus = theta
        self.q_target.transfer_weights(self.q, self.polyak_rate)

    def save(self, path, base_name):
        if not os.path.exists(path):
            os.makedirs(path)

        self.q.save(path, base_name)
        self.q_target.save(path, base_name + "_target")

        self.buffer.save(path, base_name)

        with open(os.path.join(path, base_name + "_dqn.pkl"), "wb") as f:
            pickle.dump(self.a_shape, f)
            pickle.dump(self.n_actions, f)
            pickle.dump(self.s_formatter, f)
            pickle.dump(self.n_mini_batch, f)
            pickle.dump(self.gamma, f)
            pickle.dump(self.t_cycle, f)
            pickle.dump(self.polyak_rate, f)
            pickle.dump(self.s, f)
            pickle.dump(self.a, f)
            pickle.dump(self.t, f)
            pickle.dump(self.seq_num, f)

    def load(self, path, base_name):
        self.q.load(path, base_name)
        self.q_target.load(path, base_name + "_target")

        # with open(os.path.join(path, base_name + '_rpl.npy'), 'rb') as f:
        #    self.buffer = np.load(f)

        with open(os.path.join(path, base_name + "_dqn.pkl"), "rb") as f:
            self.a_shape = pickle.load(f)
            self.n_actions = pickle.load(f)
            self.s_formatter = pickle.load(f)
            self.n_mini_batch = pickle.load(f)
            self.gamma = pickle.load(f)
            self.t_cycle = pickle.load(f)
            self.polyak_rate = pickle.load(f)
            self.s = pickle.load(f)
            self.a = pickle.load(f)
            self.t = pickle.load(f)
            self.seq_num = pickle.load(f)

    def start_state(self, s0):
        self.seq_num = 1
        self.s = self.s_formatter.convert(self.seq_num, s0)

    def next_action(self, a_sampler_fn=None):
        cur_epsilon = next(self.epsilon)
        explore = np.random.rand() < cur_epsilon
        if explore:
            # with probability epsilon select a random action a
            a = np.random.randint(self.n_actions) if a_sampler_fn is None else a_sampler_fn()
        else:
            # otherwise select a_t = argmax(Q(s_t))
            qs = self.q.predict(self.s[None, :])
            a = np.random.choice(np.flatnonzero(qs >= qs.max()))
        self.a = a
        return a

    def next_reading(self, sp, r, done, train=True):
        if not train:
            self.s = sp
            return

        s = self.s
        a = self.a
        self.seq_num += 1
        sp = self.s_formatter.convert(self.seq_num, sp, done)

        # store transition (s_t, a_t, r_t, s_{t+1}) in D
        self.buffer.store(np.r_[s, a, r, sp])

        # sample random mini-batch of transitions from D
        mini_batch_raw = self.buffer.mini_batch(self.n_mini_batch)
        states = mini_batch_raw[:, : self.s_formatter.s_shape]
        rewards = mini_batch_raw[:, self.s_formatter.s_shape + self.a_shape]
        actions = np.round(mini_batch_raw[:, self.s_formatter.s_shape]).astype(int)
        states_p = mini_batch_raw[:, -self.s_formatter.s_shape :]

        rewards_shaper = np.zeros_like(rewards)
        if self.shaper_fn is not None:
            potential_s = np.array([self.shaper_fn(s_) if not self.s_formatter.is_terminal(s_) else 0 for s_ in states])
            potential_sp = np.array([self.shaper_fn(s_) if not self.s_formatter.is_terminal(s_) else 0 for s_ in states_p])
            rewards_shaper = potential_sp * self.gamma - potential_s

        # set y_j = r_j + gamma * max(Q_hat(s_{t+1}))
        # where Q_hat(s_{t+1}) = 0 for terminal states
        actions_p = self.q_double.argmax_batch(states_p)
        q_target_values = self.q_target.predict(states_p)
        maxes = q_target_values[np.arange(self.n_mini_batch), actions_p]
        temporal_diff = rewards + rewards_shaper + self.gamma * maxes

        # replace only the slots in y for the actions that took place in each batch entry
        y = self.q.predict(states)
        y[np.arange(self.n_mini_batch), actions] = temporal_diff

        # perform a gradient descent step on (y - Q(s_j, a_j, theta))^2 with respect to parameters theta
        self.q.fit_batch(states, y)
        self.t += 1

        # every C steps, reset Q_hat <- Q
        if self.t % self.t_cycle == 0:
            self.q_target.transfer_weights(self.q, self.polyak_rate)
            self.diagnostics.record_q_swap()

        self.s = sp
