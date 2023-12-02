import time
import gensim
from gensim.models import LdaModel
from gensim import models, corpora, similarities
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt
from scipy.stats import entropy
import numpy as np
from gensim.models.callbacks import PerplexityMetric
from gensim.test.utils import common_corpus, common_dictionary


def train_lda(data, num_topics = 40, chunksize = 500, alpha ="auto", eta = "auto", passes = 25):

    num_topics = num_topics
    chunksize = chunksize
    dictionary = corpora.Dictionary(data['tokenized'])
    corpus = [dictionary.doc2bow(doc) for doc in data['tokenized']]
    t1 = time.time()
    lda = LdaModel(corpus=corpus, num_topics=40, id2word=dictionary,
                   alpha=alpha, eta=eta, chunksize=chunksize, minimum_probability=0.0, passes=passes)
    t2 = time.time()
    print("Time to train LDA model on ", data.shape[0], "articles: ", (t2-t1)/60, "min")
    return dictionary,corpus,lda

def tuning_passes(data):
    print ("Initializing with baseling: numtopic 10, chunksize 500")
    dictionary, corpus, lda = train_lda(data)
    coherences = []
    perplexities = []
    passes = []

    for i in range(25):
        ntopics, nwords = 200, 100
        if i == 0:
            p = 1
        else:
            p = i * 2
        tic = time.time()
        tuninglda = LdaModel(corpus, id2word=dictionary, num_topics=ntopics, iterations=400, passes=p)
        print('epoch', p, time.time() - tic)
        cm = CoherenceModel(model=tuninglda, corpus=corpus, coherence='u_mass')
        coherence = cm.get_coherence()
        print("Coherence", coherence)
        coherences.append(coherence)
        print('Perplexity: ', tuninglda.log_perplexity(corpus), '\n\n')
        passes.append(p)
        perplexities.append(tuninglda.log_perplexity(corpus))


    plt.plot(passes, coherences)
    plt.xlabel("Passes")
    plt.ylabel("Coherence")
    plt.show()

    plt.plot(passes, perplexities)
    plt.xlabel("Passes")
    plt.ylabel("Perplexity")
    plt.show()


def tuning_topics(data, p):
    dictionary, corpus, lda = train_lda(data)
    coherencesT = []
    perplexitiesT = []
    ntopic = []

    for i in range(20):
        if i == 0:
            ntopics = 2
        else:
            ntopics = 10 * i
        nwords = 100
        tic = time.time()
        lda4 = models.ldamodel.LdaModel(corpus, id2word=dictionary, num_topics=ntopics, iterations=400, passes=p)
        print('ntopics', ntopics, time.time() - tic)

        cm = CoherenceModel(model=lda4, corpus=corpus, coherence='u_mass')
        coherence = cm.get_coherence()
        print("Coherence", coherence)
        coherencesT.append(coherence)
        print('Perplexity: ', lda4.log_perplexity(corpus), '\n\n')
        perplexitiesT.append(lda4.log_perplexity(corpus))
        ntopic.append(ntopics)

    plt.plot(ntopic, coherencesT)
    plt.show()

    plt.plot(ntopic, perplexitiesT)
    plt.show()


def jsd(query, matrix):
    scores = []
    for i in range(0, matrix.shape[0]):
        p = query
        q = matrix.T[:, i]
        m = 0.5 * (p + q)

        jensen = np.sqrt(0.5 * (entropy(p, m)) + 0.5 * entropy(q, m))
        scores.append(jensen)

    return scores

def KL(query, matrix):
    scores = []
    for i in range(0, matrix.shape[0]):
        p = query
        q = matrix.T[:, i]


        KL = entropy(p, q)
        scores.append(KL)

    return scores