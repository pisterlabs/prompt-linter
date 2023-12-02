import numpy as np
import random
from torch.utils.data.dataloader import default_collate


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, record):  # SASR
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = record
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        return default_collate(batch)

    def __len__(self):
        return len(self.memory)


# Copied from OpenAI Baseline
class SumTree(object):
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])


# Copied from OpenAI Baseline
class PrioritizedReplayMemory(object):  # stored as ( s, a, r, s_ ) in SumTree

    e = 0.01
    a = 0.6

    def __init__(self, capacity, max_error):
        self.tree = SumTree(capacity)
        self.max_error = max_error
        self.len = 0

    def _getPriority(self, error):
        return (error + self.e) ** self.a

    def push(self, sample):  # push with maximal priority
        self.len = min(self.len + 1, self.tree.capacity)

        p = self._getPriority(self.max_error)
        self.tree.add(p, sample)

    def sample(self, batch_size):

        segment = self.tree.total() / batch_size

        index = []
        batch = []
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            index.append(idx)
            batch.append(data)

        batch = default_collate(batch)
        return index, batch

    def update(self, idx, error):
        p = self._getPriority(error)
        self.tree.update(idx, p)

    def __len__(self):
        return self.len


def copy_params(src, dst):
    params1 = src.named_parameters()
    params2 = dst.named_parameters()

    params2 = dict(params2)

    for name1, param1 in src.named_parameters():
        params2[name1].data.copy_(param1.data)


class EpsilonGreedy(object):
    def __init__(self, start_e, final_e, final_episode):
        self.start_e = start_e
        self.final_e = final_e
        self.final_episode = final_episode
        self.exp_rate = (final_e / start_e) ** (1 / final_episode)
        self.linear_rate = (start_e - final_e) / final_episode

    def linear(self, episode):
        if episode <= self.final_episode:
            return self.start_e - self.linear_rate * episode
        return self.final_e

    def exponoential(self, episode):
        if episode <= self.final_episode:
            return self.start_e * (self.exp_rate ** episode)
        return self.final_e
