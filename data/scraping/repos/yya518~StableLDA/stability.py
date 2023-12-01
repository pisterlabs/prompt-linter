# this scripts evaluate topic model stability
import numpy as np
from gensim.models import CoherenceModel
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
import itertools


class TopicModel:
    '''
    a topic model (LDA or Stable LDA) can be characterized by two probability: p(word|topic) and p(topic|doc),
    p(word|topic) also produces the top N words, and p(topic|doc) produces the topic label of a document.
    we define a TopicModel class so that we can compare two models and measure stability.
    by convention, we use theta to denote p(topic|doc) and beta to denote p(word|topic).
    bows: dataset is like the gensim's bow representation
    vocab: a list of vocab where index is the dictionary index
    '''

    def __init__(self, num_topics, theta, beta, bows, vocab):
        self.theta = theta
        self.beta = beta
        self.num_topics = num_topics
        self.bows = bows
        self.vocab = vocab
        self.num_docs = len(self.bows)

        self.get_top_n_words()
        self.get_doc_labels()
        self.get_doc_clusters()

    def get_top_n_words(self, n=10):  # return top 10 words per topic
        self.topnwords = []
        for k, prob in enumerate(self.beta):
            term_idx = np.argsort(prob)
            topnwords_k = []
            for j in np.flip(term_idx[-n:]):
                topnwords_k.append(self.vocab[j])
            self.topnwords.append(topnwords_k)
        return self.topnwords

    def print_top_n_words(self, n=10):
        # print(self.topnwords)
        for k in range(self.num_topics):
            print(' '.join(self.topnwords[k]))

    def get_doc_labels(self):  # return the document label of each doc
        self.doc_labels = []
        for idx in range(len(self.theta)):  # theta is D*K
            label = np.argmax(self.theta[idx])  # topic with max probability
            self.doc_labels.append(label)

    def get_doc_clusters(self):
        self.clusters = [set() for i in range(self.num_topics)]
        for idx in range(len(self.theta)):  # theta is D*K
            max_topic = np.argmax(self.theta[idx])  # topic with max probability
            self.clusters[max_topic].add(idx)


def computeMatrix(tm1, tm2):
    if tm1.num_topics != tm1.num_topics:
        raise ValueError('two topic models have different topics')

    matrix = np.zeros(shape=(tm1.num_topics, tm1.num_topics))
    for i in range(tm1.num_topics):
        for j in range(tm1.num_topics):
            matrix[i][j] = len(tm1.clusters[i].symmetric_difference(tm2.clusters[j]))
    return matrix


def model_alignment(tm1, tm2):
    cost_matrix = computeMatrix(tm1, tm2)
    _, alignment = linear_sum_assignment(cost_matrix)
    alignment_dict = {}
    for k in range(tm1.num_topics):
        alignment_dict[alignment[k]] = k
    return alignment_dict


def align_a_tm(tm, alignment):  ## transform a topic model based on the alignment
    old_theta, old_beta = tm.theta, tm.beta
    theta_dict, beta_dict = {}, {}
    for k in range(tm.num_topics):
        theta_dict[alignment[k]] = old_theta[:, k]
        beta_dict[alignment[k]] = old_beta[k]
    theta, beta = [], []
    for k in range(tm.num_topics):
        theta.append(theta_dict[k])
        beta.append(beta_dict[k])

    new_tm = TopicModel(tm.num_topics, np.transpose(theta), np.array(beta), tm.bows, tm.vocab)
    return new_tm


def theta_stability(tm1, tm2, alignment):  # compute document stability of two lda models,
    l1_distances = []

    for idx in range(tm1.num_docs):  # for each document, calcuate theta similarity using manhattan distance
        l1_distances.append(
            1 - 0.5 * distance.cityblock(tm1.theta[idx], [tm2.theta[idx][alignment[k]] for k in range(tm1.num_topics)]))
    return (np.mean(l1_distances))


def doc_stability(tm1, tm2, alignment):  # compute document stability of two lda models,
    index_match_bool = []
    index_match_bool = [tm1.doc_labels[idx] == alignment[tm2.doc_labels[idx]] for idx in range(tm1.num_docs)]
    return np.mean(index_match_bool)


def phi_stability(tm1, tm2, alignment):  # compute document stability of two lda models,
    l1_distances = []
    for k in range(len(alignment)):  # for each topic
        l1_distances.append(1 - 0.5 * distance.cityblock(tm2.beta[k], tm1.beta[alignment[k]]))

    return (np.mean(l1_distances))


def topwords_stability(tm1, tm2, alignment):  # compute document stability of two lda models,
    similarity = []
    for k in range(len(alignment)):  # for each topic
        sim = len(set(tm2.topnwords[k]) & set(tm1.topnwords[alignment[k]])) / float(10)
        similarity.append(sim)
    return (np.mean(similarity))

def compute_perlexity(bow, theta, phi):
    print('compute likelihood')
    loglikelihood = 0.0
    wordcount = 0
    for idx, doc in enumerate(bow):
        doc_topic = theta[idx]
        for w in doc:
            pr = 0
            wordcount += 1
            for topic, prob in enumerate(doc_topic):  # for possible topics in the document, p(t|d) * p(w|t)
                pr += prob * phi[topic][w]
            loglikelihood += np.log(pr)
    print('likelihood: %s' % loglikelihood)
    print('perplexity: %s' % np.exp(-loglikelihood / wordcount))
    return np.exp(-loglikelihood / wordcount)


def compute_coherence(gensim_bow, text, id2word, topics, coherence_score='c_npmi'):
    cm = CoherenceModel(topics=topics, corpus=gensim_bow, texts=text, dictionary=id2word, coherence=coherence_score)
    return cm.get_coherence()

def load_topic_model_results(doc_path, vocab_path, theta_path, phi_path): #load a trained topic model
    docs, vocab, theta, phi = [], [], [], []
    vocab2id = {}
    with open(vocab_path, 'r') as f:
        vocab = f.read().splitlines()
        for idx, v in enumerate(vocab):
            vocab2id[v] = idx

    with open(doc_path, 'r') as f:
        lines = f.read().splitlines()
        docs = [[vocab2id[w] for w in line.split()] for line in lines]

    with open(theta_path, 'r') as f:
        lines = f.read().splitlines()
        theta = [ [float(p) for p in line.split() ] for line in lines]
    with open(phi_path, 'r') as f:
        lines = f.read().splitlines()
        phi = [ [float(p) for p in line.split() ] for line in lines]

    return docs, vocab, theta, phi