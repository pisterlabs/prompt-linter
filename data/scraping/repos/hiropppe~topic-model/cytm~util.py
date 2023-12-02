import logging
import numpy as np
# np.seterr(all="raise")

from scipy.special import gammaln

from gensim import matutils
from gensim.corpora.dictionary import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from itertools import combinations
from tqdm import tqdm


def perplexity(L, n_kw, n_k, n_dk, n_d, alpha, beta):
    likelihood = polyad(n_dk, n_d, alpha) + polyaw(n_kw, n_k, beta)
    return np.exp(-likelihood/L)


def polyad(n_dk, n_d, alpha):
    N = n_dk.shape[0]
    K = n_dk.shape[1]
    likelihood = np.sum(gammaln(K * alpha) - gammaln(K * alpha + n_d))
    for n in range(N):
        likelihood += np.sum(gammaln(n_dk[n, :] + alpha) - gammaln(alpha))
    return likelihood


def polyaw(n_kw, n_k, beta):
    K = n_kw.shape[0]
    V = n_kw.shape[1]
    likelihood = np.sum(gammaln(V * beta) - gammaln(V * beta + n_k))
    for k in range(K):
        likelihood += np.sum(gammaln(n_kw[k, :] + beta) - gammaln(beta))
    return likelihood


def get_coherence_model(W, n_kw, top_words, coherence_model, test_texts=None, corpus=None, coo_matrix=None, coo_word2id=None, wv=None, verbose=False):

    if coo_matrix is not None:
        logging.info("Initializing PMI Coherence Model...")
        model = PMICoherence(coo_matrix, coo_word2id, W, n_kw, topn=top_words)
    elif wv is not None:
        logging.info("Initialize Word Embedding Coherence Model...")
        model = EmbeddingCoherence(wv, W, n_kw, topn=top_words)
    else:
        logging.info(f"Initializing {coherence_model} Coherence Model...")
        dictionary = Dictionary.from_documents(corpus)
        if test_texts is not None:
            model = GensimCoherenceModel(coherence_model, test_texts, None, dictionary, W, n_kw, topn=top_words, verbose=verbose)
        else:
            bow_corpus = [dictionary.doc2bow(doc) for doc in corpus]
            model = GensimCoherenceModel(coherence_model, None, bow_corpus, dictionary, W, n_kw, topn=top_words, verbose=verbose)
    return model


class GensimCoherenceModel():

    def __init__(self, model, texts, corpus, dictionary, W, n_kw, topn=20, verbose=False):
        self.model = model
        self.texts = texts
        self.corpus = corpus
        self.dictionary = dictionary
        self.W = W
        self.n_kw = n_kw
        self.topn = topn
        self.K = len(n_kw)
        self.verose = verbose

    def get_topics(self):
        topics = []
        for k in range(self.K):
            topn_indices = matutils.argsort(self.n_kw[k], topn=self.topn, reverse=True)
            topics.append([self.W[w] for w in topn_indices])
        return topics

    def score(self):
        topics = self.get_topics()
        if self.model == 'u_mass':
            cm = CoherenceModel(topics=topics,
                                corpus=self.corpus, dictionary=self.dictionary, coherence=self.model)
        else:
            cm = CoherenceModel(topics=topics,
                                texts=self.texts, dictionary=self.dictionary, coherence=self.model)
        if self.verose:
            coherences = cm.get_coherence_per_topic()
            for index, topic in enumerate(topics):
                print(str(index) + ':' + str(coherences[index]) + ':' + ','.join(topic))
        return cm.get_coherence()


class EmbeddingCoherence():

    def __init__(self, wv, W, n_kw, topn=20):
        self.wv = wv
        self.W = W
        self.n_kw = n_kw
        self.topn = topn
        self.K = len(n_kw)

    def score(self):
        scores = []
        for k in range(self.K):
            topn_indices = matutils.argsort(self.n_kw[k], topn=self.topn, reverse=True)
            for x, y in combinations(topn_indices, 2):
                w_x, w_y = self.W[x], self.W[y]
                if w_x in self.wv and w_y in self.wv:
                    scores.append(self.wv.similarity(w_x, w_y))
        return np.mean(scores)


class PMICoherence():

    def __init__(self, M, word2id, W, n_kw, eps=1e-08, topn=20):
        self.M = M
        self.M.setdiag(0)
        self.word2id = word2id
        self.W = W
        self.n_kw = n_kw
        self.eps = eps
        self.topn = topn
        self.K = len(n_kw)
        self.N = np.sum(M)

        V = len(W)
        self.n_w = np.zeros((V), dtype=np.int32)
        for i in tqdm(range(V)):
            if W[i] in word2id:
                self.n_w[i] = self.M[:, word2id[W[i]]].sum()
            else:
                self.n_w[i] = 0

    def pmi(self, x, y, w_x, w_y):
        ix = self.word2id[w_x]
        iy = self.word2id[w_y]
        X = self.n_w[x]
        Y = self.n_w[y]
        XY = self.M[ix, iy]
        if XY == 0 or X == 0 or Y == 0:
            pmi = 0
        else:
            # pmi = np.log2(XY*N/(X*Y+self.eps))/(-np.log(XY/self.N) + self.eps)
            p_xy = XY/self.N
            p_x = X/self.N
            p_y = Y/self.N
            pmi = np.log2(p_xy/(p_x*p_y+self.eps))/(-np.log(p_xy) + self.eps)
        return pmi

    def score(self):
        scores = []
        for k in range(self.K):
            topn_indices = matutils.argsort(self.n_kw[k], topn=self.topn, reverse=True)
            for x, y in combinations(topn_indices, 2):
                w_x, w_y = self.W[x], self.W[y]
                if w_x in self.word2id and w_y in self.word2id:
                    scores.append(self.pmi(x, y, w_x, w_y))
        return np.mean(scores)
