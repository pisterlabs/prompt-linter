from gensim import corpora
from geneticalgorithm2 import geneticalgorithm2 as ga
from geneticalgorithm2 import Crossover, Mutations, Selection
import time,pickle, numpy as np
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel

def main():
    start_time = time.time()
    file = open('preprocessing\\preprocessedData.txt', 'rb')
    texts = pickle.load(file)
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    # ga
    pop_size = 10
    crossover_prob = 0.6
    mutation_prob = 0.01


    def fitness(c):
        print(c)
        ldamodel = LdaModel(corpus=corpus, id2word=dictionary, num_topics=int(c[0]), alpha=float(c[2]), eta=float(c[2]))
        coherence = CoherenceModel(model=ldamodel, texts=texts, dictionary=dictionary, coherence= 'c_v')
        return -(coherence.get_coherence())


    varbounds = np.array([[2, 10], [0, 1], [0, 1]])
    var_types = np.array(['int', 'real', 'real'])
    alg_param = {'max_num_iteration': 100,
                'population_size': pop_size,
                'mutation_probability': mutation_prob,
                'elit_ratio': 0.2,
                'parents_portion' : 0.2,
                'crossover_probability': crossover_prob,
                'crossover_type': Crossover.arithmetic(),
                'mutation_type': Mutations.uniform_by_center(),
                'selection_type': Selection.roulette(),
                'max_iteration_without_improv': 10}

    model = ga(fitness, 3, function_timeout=20.0 , variable_type_mixed=var_types, variable_boundaries=varbounds, algorithm_parameters=alg_param)
    model.run()

    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    main()