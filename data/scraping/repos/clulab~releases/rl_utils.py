"""
Most of the code is from https://github.com/Curt-Park/rainbow-is-all-you-need
There were some modifications to make it suitable for our usecase
"""

import collections
from queryast import AstNode

import numpy as np
import random
import torch

# -*- coding: utf-8 -*-
"""Segment tree for Prioritized Replay Buffer."""

import operator
from typing import Callable, Dict, Iterator, List


class ProblemSpecification:
    def __init__(self, sentences: List[List[str]], specs: List[dict], vocabulary: dict) -> None:
        self.sentences = sentences
        self.specs = specs
        self.vocabulary = {x:sorted(list(set(vocabulary[x]))) for x in vocabulary}

    def __str__(self) -> str:
        return f'ProblemSpecification(sentences: {self.sentences}, specs: {self.specs}, vocabulary: {self.vocabulary})'
        
    def construct_query(self, query: str):
        return {
            'query': query,
            'sentences': self.sentences,
            'specs': self.specs,
        }

    def hash(self):
        hashes = []
        for spec in self.specs:
            sentence = self.sentences[spec['sentId']]
            start    = spec['start']
            end      = spec['end']
            string_to_hash = ' '.join(sentence) + f' {start} {end}'
            hashes.append(hash(string_to_hash))

        return sum(hashes)


class OdinsynthEnvStep:
    def __init__(self, query: AstNode, problem_specification: ProblemSpecification):
        self.query = query
        self.problem_specification = problem_specification
    
    def __str__(self):
        return f"{self.query.pattern()} - {self.problem_specification}"

    def hash(self):
        return hash(self.query.pattern()) + self.problem_specification.hash()



class SegmentTree:
    """ Create SegmentTree.

    Taken from OpenAI baselines github repository:
    https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py

    Attributes:
        capacity (int)
        tree (list)
        operation (function)

    """

    def __init__(self, capacity: int, operation: Callable, init_value: float):
        """Initialization.

        Args:
            capacity (int)
            operation (function)
            init_value (float)

        """
        assert (
            capacity > 0 and capacity & (capacity - 1) == 0
        ), "capacity must be positive and a power of 2."
        self.capacity = capacity
        self.tree = [init_value for _ in range(2 * capacity)]
        self.operation = operation

    def _operate_helper(
        self, start: int, end: int, node: int, node_start: int, node_end: int
    ) -> float:
        """Returns result of operation in segment."""
        if start == node_start and end == node_end:
            return self.tree[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._operate_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._operate_helper(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                return self.operation(
                    self._operate_helper(start, mid, 2 * node, node_start, mid),
                    self._operate_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end),
                )

    def operate(self, start: int = 0, end: int = 0) -> float:
        """Returns result of applying `self.operation`."""
        if end <= 0:
            end += self.capacity
        end -= 1

        return self._operate_helper(start, end, 1, 0, self.capacity - 1)

    def __setitem__(self, idx: int, val: float):
        """Set value in tree."""
        idx += self.capacity
        self.tree[idx] = val

        idx //= 2
        while idx >= 1:
            self.tree[idx] = self.operation(self.tree[2 * idx], self.tree[2 * idx + 1])
            idx //= 2

    def __getitem__(self, idx: int) -> float:
        """Get real value in leaf node of tree."""
        assert 0 <= idx < self.capacity

        return self.tree[self.capacity + idx]


class SumSegmentTree(SegmentTree):
    """ Create SumSegmentTree.

    Taken from OpenAI baselines github repository:
    https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py

    """

    def __init__(self, capacity: int):
        """Initialization.

        Args:
            capacity (int)

        """
        super(SumSegmentTree, self).__init__(
            capacity=capacity, operation=operator.add, init_value=0.0
        )

    def sum(self, start: int = 0, end: int = 0) -> float:
        """Returns arr[start] + ... + arr[end]."""
        return super(SumSegmentTree, self).operate(start, end)

    def retrieve(self, upperbound: float) -> int:
        """Find the highest index `i` about upper bound in the tree"""
        # TODO: Check assert case and fix bug
        assert 0 <= upperbound <= self.sum() + 1e-5, "upperbound: {}".format(upperbound)

        idx = 1

        while idx < self.capacity:  # while non-leaf
            left = 2 * idx
            right = left + 1
            if self.tree[left] > upperbound:
                idx = 2 * idx
            else:
                upperbound -= self.tree[left]
                idx = right
        return idx - self.capacity


class MinSegmentTree(SegmentTree):
    """ Create SegmentTree.

    Taken from OpenAI baselines github repository:
    https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py

    """

    def __init__(self, capacity: int):
        """Initialization.

        Args:
            capacity (int)

        """
        super(MinSegmentTree, self).__init__(
            capacity=capacity, operation=min, init_value=float("inf")
        )

    def min(self, start: int = 0, end: int = 0) -> float:
        """Returns min(arr[start], ...,  arr[end])."""
        return super(MinSegmentTree, self).operate(start, end)


class ReplayBuffer:
    """
    A simple replay buffer.
    Consists of a list instead of a numpy/tensor because
    our type of data is not numeric and we do the encoding
    in the model
    """

    def __init__(self, size: int, metadata: dict = {}):
        self.obs_buf = [None] * size
        self.next_obs_buf = [None] * size
        self.acts_buf = [None] * size
        self.rews_buf = [None] * size
        self.done_buf = [None] * size
        self.max_size = size
        self.ptr, self.size, = 0, 0

    def store(
        self,
        obs,
        act, 
        rew, 
        next_obs, 
        done,
    ):
        self.obs_buf[self.ptr] = obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.next_obs_buf[self.ptr] = next_obs
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    """
    Update the internals of this replay buffer according to its internal update policy
    There is no update policy in this case. This type of replay buffer is a FIFO type with
    a max capacity
    """
    def update(self, data: dict):
        return


    """
    Sample from the underlying lists
    Uses a metadata parameter to make the signature uniform among
    other replay buffers which need additional parameters
    """
    def sample(self, batch_size, metadata = {}) -> Dict[str, np.ndarray]:
        idxs = np.random.choice(self.size, size=batch_size, replace=False)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
        
        for idx in idxs:
            s_lst.append(self.obs_buf[idx])
            a_lst.append(self.acts_buf[idx])
            r_lst.append(self.rews_buf[idx])
            s_prime_lst.append(self.next_obs_buf[idx])
            done_mask_lst.append(self.done_buf[idx])

        return ExperienceReplayBatch(
            s_lst, 
            a_lst,
            r_lst, 
            s_prime_lst,
            done_mask_lst,
            )
               
    def __len__(self) -> int:
        return self.size



class PrioritizedReplayBuffer(ReplayBuffer):
    """Prioritized Replay buffer.
    
    Attributes:
        max_priority (float): max priority
        tree_ptr (int): next index of tree
        alpha (float): alpha parameter for prioritized replay buffer
        sum_tree (SumSegmentTree): sum tree for prior
        min_tree (MinSegmentTree): min tree for min prior to get max weight
        
    """
    
    def __init__(
        self, 
        size: int, 
        metadata: dict = {'alpha': 0.6, 'total_timesteps': 10000},
    ):
        """Initialization."""
        assert metadata['alpha'] >= 0
        
        super(PrioritizedReplayBuffer, self).__init__(size)
        self.max_priority, self.tree_ptr = 1.0, 0
        self.metadata = metadata
        self.alpha = metadata['alpha']
        self.beta = metadata['beta']
        self.total_timesteps  = metadata['total_timesteps']
        
        # capacity must be positive and a power of 2.
        tree_capacity = 1
        while tree_capacity < self.max_size:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)

        self.global_timestep = 0
        
    def store(
        self, 
        obs: OdinsynthEnvStep,
        act: int, 
        rew: float, 
        next_obs: OdinsynthEnvStep,
        done: bool
    ):
        """Store experience and priority."""
        super().store(obs, act, rew, next_obs, done)
        
        self.sum_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.min_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.tree_ptr = (self.tree_ptr + 1) % self.max_size
        
    """
    Sample from the underlying lists
    Uses a metadata parameter to access the beta parameter used 
    for sampling
    """
    def sample(self, batch_size) -> Dict[str, np.ndarray]:
        self.global_timestep += 1

        """Sample a batch of experiences."""
        assert len(self) >= batch_size

        fraction = min(self.global_timestep / self.total_timesteps, 1.0)
        self.beta = self.beta + fraction * (1.0 - self.beta)
        assert self.beta > 0

        indices = self._sample_proportional(batch_size)
        
        obs = [self.obs_buf[i] for i in indices]
        next_obs = [self.next_obs_buf[i] for i in indices]
        acts = [self.acts_buf[i] for i in indices]
        rews = [self.rews_buf[i] for i in indices]
        done = [self.done_buf[i] for i in indices]
        weights = [self._calculate_weight(i, self.beta) for i in indices]
        return PrioritizedExperienceReplayBatch(obs, acts, rews, next_obs, done, weights, indices)
        
    def update_priorities(self, indices: List[int], priorities: np.ndarray):
        """Update priorities of sampled transitions."""
        assert len(indices) == len(priorities)

        for idx, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= idx < len(self)

            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha

            self.max_priority = max(self.max_priority, priority)


    """
    Update the internals of this replay buffer according to its internal update policy
    The update policy in this case is to assign a priority (weight) to each example
    Update the priorities accordingly
    """
    def update(self, data: dict):
        new_priorities = data['loss_for_prior'] + data['prior_eps']
        self.update_priorities(data['indices'], new_priorities)


    def _sample_proportional(self, batch_size) -> List[int]:
        """Sample indices based on proportions."""
        indices = []
        p_total = self.sum_tree.sum(0, len(self) - 1)
        segment = p_total / batch_size
        
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)
            
        return indices
    
    def _calculate_weight(self, idx: int, beta: float):
        """Calculate the weight of the experience at idx."""
        # get max weight
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * len(self)) ** (-beta)
        
        # calculate weights
        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * len(self)) ** (-beta)
        weight = weight / max_weight
        
        return weight


ExperienceReplayBatch            = collections.namedtuple('ExperienceReplayBatch',            ['obs', 'acts', 'rews', 'next_obs', 'done']) 
PrioritizedExperienceReplayBatch = collections.namedtuple('PrioritizedExperienceReplayBatch', ['obs', 'acts', 'rews', 'next_obs', 'done', 'weights', 'indices']) 


# NOTE The batching is handled here; That is, when iterating
# over this dataset you obtain a batch (with batch_size specified
# when creating the object). This class inherits from IterableDataset
# because the underlying buffer (@param buffer) can grow in size
# during training
class ReplayBufferDataset(torch.utils.data.IterableDataset):
    def __init__(self, buffer, batch_size, total_timesteps) -> None:
        super(ReplayBufferDataset).__init__()
        self.batch_size = batch_size
        self.buffer = buffer
        self.total_timesteps = total_timesteps

    def __iter__(self) -> Iterator:
        while True:
            sample = self.buffer.sample(self.batch_size)
            yield sample


# A pytorch dataset over a list
class ListDataset(torch.utils.data.Dataset):
    def __init__(self, data: List) -> None:
        super(ListDataset).__init__()
        self.data = data

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]



import gym
class ListBasedGymEnv(gym.Wrapper):
    def __init__(self, env=None):
        super().__init__(env)

    def step(self, action):
        step = self.env.step(action)
        return (step[0].tolist(),) + step[1:]

    def reset(self):
        res = self.env.reset()
        return res.tolist()

# per = PrioritizedReplayBuffer(300, 1000, 3, 0.6)
# per.store(['a', 'b'], 0, 0.1, ['a', 'c'], False)
# per.store(['a', 'b'], 1, 0.2, ['a', 'd'], False)
# per.store(['b', 'a'], 0, 0.3, ['b', 'd'], False)
# per.store(['b', 'a'], 1, 0.3, ['b', 'e'], False)
# per.store(['a', 'a'], 0, 0.3, ['b', 'b'], False)
# per.store(['a', 'a'], 1, 0.3, ['a', 'c'], False)
# x = per.sample_batch()


