#code from openai
#https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py

import numpy as np
import random

class ReplayBuffer(object):
    def __init__(self, size):
        """Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 998400

    def __len__(self):
        return len(self._storage)

    def push(self, state, action, aggregated_cost, costs, next_state, goal, done, constraints):
        data = (state, action, aggregated_cost, costs, next_state, goal, done, constraints)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = int((self._next_idx + 1) % self._maxsize)

    def _encode_sample(self, idxes):
        obses_t, actions, costs_aggregated, costs_list, obses_tp1, goals, dones, constraints = [], [], [], [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, aggregated_cost, costs, obs_tp1, goal, done, constraint = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            costs_aggregated.append(aggregated_cost)
            costs_list.append(np.array(costs, copy=False))
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
            goals.append(np.array(goal, copy=False))
            constraints.append(np.array(constraint, copy=False))
        return np.array(obses_t), np.array(actions), np.array(costs_aggregated), np.array(costs_list), np.array(obses_tp1), np.array(goals), np.array(dones), np.array(constraints)

    def sample(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        agg_cost_batch: np.array
            aggregated costs received as results of executing act_batch
        costs_batch: np.array
            costs received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        goals_batch: np.array
            targeted goals
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        constraint_batch: np.array
            whether constraints expressed in goals are violated for states in obs_batch.
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)



