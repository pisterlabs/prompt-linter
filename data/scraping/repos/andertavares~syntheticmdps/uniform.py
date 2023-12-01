import scipy.stats as stats
import numpy as np


class UniformAgent(object):
    def __init__(self, bandit):
        self.bandit = bandit

        # assigns uniformly random probs. to arms, then normalize
        self.probabilities = stats.uniform(0, 1).rvs(len(bandit.arms))

        sum_probs = sum(self.probabilities)

        # normalizes
        self.probabilities = [p / sum_probs for p in self.probabilities]

        #checks
        assert np.isclose(1, sum(self.probabilities))

        #saves cumulative probabilities:
        self.cumprobs = np.cumsum(self.probabilities)

    def act(self):
        """
        Selects an arm in proportion with the probabilities
        """
        # code copied from OpenAI Gym gym/envs/toy_text/discrete.py
        return (self.cumprobs > np.random.rand()).argmax()

    def argmax(self):
        """
        Returns the arm with highest probability
        """
        return np.argmax(self.probabilities)

    def epsilon_argmax(self, epsilon):
        if np.random.random() < epsilon:
            return np.random.randint(0, len(self.probabilities))
        return self.greedy()

    def learn(self, *args):
        pass

    def __str__(self):
        return "UniformAgent"


class UniformBiasedAgent(UniformAgent):
    def __init__(self, bandit):
        self.bandit = bandit

        # samples the prob of choosing the best arm from [0, 1]
        p_best = stats.uniform(0, 1).rvs(1)[0]

        # samples the prob of choosing the other arms from [0, 1-p_best]
        p_others = stats.uniform(0, 1 - p_best).rvs(len(bandit.arms) -1)

        # normalizes
        sum_others = sum(p_others)
        #norm_factor = sum_others*(1 - p_best)
        p_others = [p *(1 - p_best) / sum_others for p in p_others]

        #print(p_best, sum_others, sum(p_others))

        # finally assigns the probabilities
        offset = 0 # helps on getting the prob. from correct position
        self.probabilities = np.zeros(len(bandit.arms))
        for i, arm in enumerate(bandit.arms):
            if i == bandit.best_arm:
                self.probabilities[i] = p_best
                offset = 1 # to discount that I'm not getting p_others[i] in this iteration
            else:
                self.probabilities[i] = p_others[i - offset]

        #checks
        assert np.isclose(1, sum(self.probabilities))

        #saves cumulative probabilities:
        self.cumprobs = np.cumsum(self.probabilities)

    def __str__(self):
        return "UniformBiasedAgent"