import numpy as np
from random import choice
import matplotlib.pyplot as plt

import pickle

from random import random
from segment_tree import SumSegmentTree

def discretize(bd, grid):
    return grid[min(np.digitize(bd, grid), len(grid)-1)]

'''
Adapted from OpenAI RL baseline:
    https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
'''
class Archive:
    '''
    dims: [(dim1_low, dim1_high, dim1_step) * len(dims)]

    priority_buffer_alpha: how strongly to prefer sampling high fitness
        individuals (range is [0,1] where 0 means fair sampling)
    '''
    def __init__(self, dims, priority_buffer_alpha=0.9):
        self.dims = [np.arange(d[0], d[1], d[2]) for d in dims]
        self.archive = {}

        self._maxsize = np.prod([len(d) for d in self.dims])
        self._storage = []

        it_capacity = 1
        while it_capacity < self._maxsize:
            it_capacity *= 2
        self._it_sum = SumSegmentTree(it_capacity)

        self._next_idx = 0
        self._alpha = priority_buffer_alpha

        # FIXME: sumtree needs positive entries
        self.mean_sum = 1

    def add(self, individual, parent=None):
        discretized_bds = \
            list(map(lambda bd, grid: discretize(bd, grid), individual.bds, self.dims))
        individual.discretized_bds = discretized_bds

        idx = None
        # if self._storage does NOT contain the new individual's discretized_bds,
        #   append the new individual and increment self._next_idx
        if not individual in self.archive:
            # update the parent in sumtree for discovering a new individual
            if parent is not None:
                for i, p in enumerate(self._storage):
                    if p == parent:
                        self.mean_sum = (self.mean_sum*self._next_idx - \
                                            self._it_sum[i] + \
                                            individual.fitness**self._alpha) / \
                                            (self._next_idx + 1)
                        self._it_sum[i] = individual.fitness ** self._alpha
                        break

            # add new individual to archive and sumtree
            #   sumtree ranks by ability to discover new and better individuals
            #   so initialize with mean_sum since we don't yet know how the
            #   new individual performs in this regard
            self.archive[individual] = individual
            self._storage.append(individual)
            self._it_sum[self._next_idx] = self.mean_sum
            self._next_idx += 1
            return individual.fitness ** self._alpha
        # if self._storage already contains the new individual's discretized_bds,
        #   replace the old individual with the new individual
        else:
            for i, idv in enumerate(self._storage):
                if idv == individual:
                    if individual.fitness > idv.fitness:
                        # parent discovered better individual, update its sumtree
                        #   item with fitness improvement
                        if parent is not None:
                            for ip, p in enumerate(self._storage):
                                if p == parent:
                                    self.mean_sum = (self.mean_sum*self._next_idx - \
                                                        self._it_sum[ip] + \
                                                        (individual.fitness-idv.fitness)**self._alpha) / \
                                                        self._next_idx
                                    self._it_sum[ip] = (individual.fitness-idv.fitness)**self._alpha
                                    break

                        # update the archive with the individual just discovered
                        #   no need to update sumtree for new individual because
                        #   there's no information on its ability to discover new
                        #   and better individuals
                        self.archive[individual] = individual
                        self._storage[i] = individual
                        return (individual.fitness-idv.fitness)**self._alpha
                    else:
                        # parent doesn't discover better individual, reset its
                        # sumtree item to mean_sum
                        if parent is not None:
                            for ip, p in enumerate(self._storage):
                                if p == parent:
                                    self._it_sum[ip] = self.mean_sum
                                    break
                        return 0

    def sample(self, batch_size):
        return np.random.choice(list(self.archive.values()), batch_size)

    def importance_sample(self, batch_size):
        res = []
        p_total = self._it_sum.sum(0, len(self._storage) - 1)
        every_range_len = p_total / batch_size
        for i in range(batch_size):
            mass = random() * every_range_len + i * every_range_len
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)

        return [self._storage[i] for i in res]

    def find_best(self):
        max_fitness = float('-inf')
        best_idv = None
        for idv in self.archive.values():
            if idv.fitness > max_fitness:
                max_fitness = idv.fitness
                best_idv = idv

        return best_idv

    @staticmethod
    def get_best_2D(archive, dim1, dim2):
        result = None
        best_fitness = float('-inf')
        for idv in archive.archive.values():
            if np.isclose(idv.discretized_bds[0], dim1) and np.isclose(idv.discretized_bds[1], dim2):
                if idv.fitness > best_fitness:
                    result = idv
                    best_fitness = idv.fitness
        return result

    @staticmethod
    def visualize(archive2D):
        dim1 = archive2D.dims[0].tolist()
        dim1 += [dim1[-1]+dim1[-1]-dim1[-2]]
        dim2 = archive2D.dims[1].tolist()
        dim2 += [dim2[-1]+dim2[-1]-dim2[-2]]
        fitness_mat = np.zeros((len(dim1), len(dim2)))
        for idv in archive2D.archive.values():
            index1 = np.where(idv.discretized_bds[0] == archive2D.dims[0])[0]
            index2 = np.where(idv.discretized_bds[1] == archive2D.dims[1])[0]
            if fitness_mat[index1, index2] < idv.fitness:
                fitness_mat[index1, index2] = idv.fitness

        z_min = np.min(fitness_mat)
        z_max = max(np.max(fitness_mat), -z_min)
        fig, ax = plt.subplots()
        c = ax.pcolormesh(dim1, dim2, fitness_mat, cmap='RdBu', vmin=z_min, vmax=z_max)
        ax.axis([np.min(dim1), np.max(dim1), np.min(dim2), np.max(dim2)])
        fig.colorbar(c, ax=ax)

        plt.show()

    @classmethod
    def from_pickle(cls, filename):
        try:
            return pickle.load(open(filename, 'rb'))
        except FileNotFoundError:
            print('Cannot find filename, exiting...')
            quit()
        except Exception as err:
            print(err)
            quit()

    def save(self, filename):
        pickle.dump(self, open(filename, 'wb'))
