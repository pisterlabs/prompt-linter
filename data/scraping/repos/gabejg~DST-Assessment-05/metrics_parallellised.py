from multiprocessing import pool, cpu_count, current_process
import datetime

import pandas as pd
import numpy as np
import pickle

import gensim
from gensim.models.coherencemodel import CoherenceModel

def compute_values(num_topics,dictionary,bow_corpus,processed_text):

    model = gensim.models.LdaMulticore(bow_corpus, num_topics = num_topics, id2word=dictionary, workers = 4)

    coherence_model_lda = CoherenceModel(model=model,corpus=bow_corpus, dictionary=dictionary, coherence='u_mass')
    coherence_lda = coherence_model_lda.get_coherence()

    coherence_model_lda2 = CoherenceModel(model=model,texts = processed_text, dictionary=dictionary, coherence='c_v')
    coherence_lda2 = coherence_model_lda2.get_coherence()

    scores = []
    scores.append(coherence_lda)
    scores.append(coherence_lda2)
    scores.append(lda_model.log_perplexity(bow_corpus))

    return scores

if __name__ == "__main__":

    begin_time = datetime.datetime.now()

    # print the number of cores
    print("Number of cores available equals %s" % cpu_count())
    print("Using %s cores" % cpu_count())

    dictionary = pickle.load(open('../Data/dictionary.p','rb'))
    bow_corpus = pickle.load(open('../Data/bow_corpus.p','rb'))
    processed_text = pickle.load(open('../Data/processed_text.p','rb'))

    iterable = list(zip(range(2,40,2),repeat(dictionary),repeat(bow_corpus),repeat(processed_text)))

    # create a pool of workers
    # start all worker processes
    pool = Pool(processes = cpu_count())

    with pool as p:
        results = pool.starmap(compute_values, iterable)

    print("Finished calculating scores!")
    end_time = datetime.datetime.now()
    print(end_time - begin_time)
