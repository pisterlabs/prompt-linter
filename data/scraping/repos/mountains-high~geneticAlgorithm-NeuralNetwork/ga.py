import random
from openAI import openAI
import numpy as np
from population import Individue, Population


class GA:

    def __init__(self, mutation_rate, crossover_rate, eval):
        '''Define parameters and create a population of N elements, each randomly generated'''
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.openAI = openAI(eval)

    def selection(self, pop):
        '''Evaluate the fitness of each element of the population and build a matting pool'''
        for i in range(int(len(pop) / 2)):
            parents = []
            for j in range(2):
                first_candidate_index = random.randint(1, len(pop) - 1)
                second_candidate_index = random.randint(1, len(pop) - 1)
                first_candidate_fitness = pop[first_candidate_index].fitness
                second_candidate_fitness = pop[second_candidate_index].fitness
                if first_candidate_fitness > second_candidate_fitness:
                    parents.append(pop[first_candidate_index])
                else:
                    parents.append(pop[second_candidate_index])
            return parents

    def crossover_(self, parents):
        '''
        Create a child combining the genotype of two parents
        For each weight matrix, W1, W2, W3 crossover individually
        '''

        # For W1:
        w1_first_child = np.concatenate([parents[0].weights[0][:2, :], parents[1].weights[0][2:, 0:]])
        w1_second_child = np.concatenate([parents[1].weights[0][:2, :], parents[0].weights[0][2:, 0:]])
        # np.random.shuffle(w1_first_child)
        # np.random.shuffle(w1_second_child)
        # For W2:
        w2_first_child = np.concatenate([parents[0].weights[1][:2, :], parents[1].weights[1][2:, 0:]])
        w2_second_child = np.concatenate([parents[1].weights[1][:2, :], parents[0].weights[1][2:, 0:]])
        # np.random.shuffle(w2_first_child)
        # np.random.shuffle(w2_second_child)
        # For W3:
        w3_first_child = np.concatenate([[parents[0].weights[2][0]], [parents[1].weights[2][1]]])
        w3_second_child = np.concatenate([[parents[1].weights[2][0]], [parents[0].weights[2][1]]])
        # np.random.shuffle(w3_first_child)
        # np.random.shuffle(w3_second_child)
        first_child = Individue()
        second_child = Individue()
        first_child.weights = [w1_first_child, w2_first_child, w3_first_child]
        # first_child.weights = [w1_second_child, w2_second_child, w3_second_child]
        return [first_child, second_child]

    def crossover(self, parents):
        '''
        Create a child combining the genotype of two parents
        For each weight matrix, W1, W2, W3 crossover individually
        '''
        W = []
        for w in range(3):
            child = np.mean(np.array([parents[0].weights[w], parents[1].weights[w]]), axis=0)
            W.append(child)
        first_child = Individue()
        second_child = Individue()
        # first_child.weights = W
        return [first_child, second_child]

    def mutation_(self, individues):
        if random.random() <= self.mutation_rate:
            # For W1:
            first_row_index, second__row_index = random.sample(range(4), 2)
            first_col_index, second__col_index = random.sample(range(4), 2)
            aux = individues[0].weights[0][first_row_index][first_col_index]
            individues[0].weights[0][first_row_index][first_col_index] = individues[0].weights[0][second__row_index][
                second__col_index]
            individues[0].weights[0][second__row_index][second__col_index] = aux
            # For W2:
            first_col_index, second__col_index = random.sample(range(4), 2)
            aux = individues[0].weights[1][first_col_index][0]
            individues[0].weights[1][first_col_index][0] = individues[0].weights[1][first_col_index][1]
            individues[0].weights[1][first_col_index][1] = aux
            # For W3:
        return individues

    def mutation(self, individues):
        if random.random() <= self.mutation_rate:
            for child in individues:
                # For W1:
                np.random.shuffle(child.weights[0])
                # For W2:
                np.random.shuffle(child.weights[1])
                # For W3:
                np.random.shuffle(child.weights[2])
        return individues

    def calculate_fitness(self, individue):
        return self.openAI.run_simulation(individue)

    def generation(self, population):
        '''
        1. Selection: Pick two parents with probability according to relative fitness
        2. Crossover: Create a child combining the genotype of these two parents
        3. Mutation: Mutate the child based on a given probability
        4. Add the new child to a new population
        '''
        new_population = Population(0)
        for i in range(int(len(population) / 2)):
            parents = self.selection(population)
            if random.random() <= self.crossover_rate:
                children = self.crossover(parents)
            else:
                children = parents
            children = self.mutation(children)
            for child in children:
                new_population.add(child)
        return new_population
