import math
import random
from gensim import corpora
from ..helpers.exceptions import MissingModelError, MissingDataSetError
from ..wrappers import LDAMallet, TNDMallet
from gensim.models import CoherenceModel


class NLDA:
    '''
    Noiseless Latent Dirichlet Allocation (NLDA).
    An ensemble topic-noise model consisting of the noise distribution from
    TND and the topic-word distribution from LDA.
    Input the raw data and compute the whole model, or input pre-computed distributions
    for faster inference.

    :param dataset: list of lists, required.
    :param tnd_k: int, optional:
        Number of topics to compute in TND.
    :param tnd_alpha: int, optional:
            Alpha parameter of TND.
    :param tnd_beta0: float, optional:
            Beta_0 parameter of TND.
    :param tnd_beta1: int, optional
            Beta_1 (skew) parameter of TND.
    :param tnd_noise_words_max: int, optional:
            Number of noise words to save when saving the distribution to a file.
            The top `noise_words_max` most probable noise words will be saved.
    :param tnd_iterations: int, optional:
            Number of training iterations for TND.
    :param lda_iterations: int, optional:
            Number of training iterations for LDA.
    :param lda_k: int, optional:
        Number of topics to compute in LDA.
    :param phi: int, optional:
        Topic weighting for noise filtering step.
    :param topic_depth: int, optional:
        Number of most probable words per topic to consider for replacement in noise filtering step.
    :param top_words: int, optional:
        Number of words per topic to return.
    :param tnd_noise_distribution: dict, optional:
        Pre-trained noise distribution
    :param lda_tw_dist: dict, optional:
        Pre-trained topic-word distribution.
    :param lda_topics: list of lists, optional:
        Pre-computed LDA topics.
    :param corpus: Gensim object, optional:
        Formatted documents for use in model.  Automatically computed if not provided.
    :param dictionary: Gensim object, optional:
        Formatted word mapping for use in model.  Automatically computed if not provided.
    :param mallet_tnd_path: path to Mallet TND code, required:
        Path should be `path/to/mallet-tnd/bin/mallet`.
    :param mallet_lda_path: path to Mallet LDA code, required:
        Path should be `path/to/mallet-lda/bin/mallet`.
    :param random_seed: int, optional:
        Seed for random-number generated processes.
    :param run: bool, optional:
        If true, run model on initialization, if data is provided.
    :param tnd_workers: int, optional:
        Number of cores to use for computation of TND.
    :param lda_workers: int, optional:
        Number of cores to use for computation of LDA.
    '''

    def __init__(self, dataset=None, tnd_k=30, tnd_alpha=50, tnd_beta0=0.01, tnd_beta1=25, tnd_noise_words_max=200,
                 tnd_iterations=1000, lda_iterations=1000, lda_k=30, phi=10, topic_depth=100, top_words=20,
                 tnd_noise_distribution=None, lda_tw_dist=None, lda_topics=None, corpus=None, dictionary=None,
                 mallet_tnd_path=None, mallet_lda_path=None, random_seed=1824, run=True,
                 tnd_workers=4, lda_workers=4):
        self.topics = []
        self.dataset = dataset
        self.tnd_k = tnd_k
        self.tnd_alpha = tnd_alpha
        self.tnd_beta0 = tnd_beta0
        self.tnd_beta1 = tnd_beta1
        self.tnd_noise_words_max = tnd_noise_words_max
        self.tnd_iterations = tnd_iterations
        self.lda_iterations = lda_iterations
        self.lda_k = lda_k
        self.nlda_phi = phi
        self.nlda_topic_depth = topic_depth
        self.top_words = top_words
        self.tnd_noise_distribution = tnd_noise_distribution
        self.lda_tw_dist = lda_tw_dist
        self.lda_topics = lda_topics
        self.corpus = corpus
        self.dictionary = dictionary
        self.mallet_tnd_path = mallet_tnd_path
        self.mallet_lda_path = mallet_lda_path
        self.random_seed = random_seed
        random.seed(self.random_seed)
        self.lda_workers = lda_workers
        self.tnd_workers = tnd_workers

        if self.mallet_tnd_path is None and self.tnd_noise_distribution is None:
            raise MissingModelError('tnd')
        if self.mallet_lda_path is None and self.lda_tw_dist is None:
            raise MissingModelError('lda')
        if self.dataset is None and (self.corpus is None or self.dictionary is None
                                     or self.tnd_noise_distribution is None or self.lda_tw_dist is None):
            raise MissingDataSetError

        if run:
            if (self.dataset is not None) and (self.corpus is None or self.dictionary is None):
                self._prepare_data()
            if self.tnd_noise_distribution is None:
                self._compute_tnd()
            if self.lda_tw_dist is None:
                self._compute_lda()

            self._compute_nlda()

    def _prepare_data(self):
        """
        takes dataset, sets self.dictionary and self.corpus for use in Mallet models and NLDA

        :return: void
        """
        dictionary = corpora.Dictionary(self.dataset)
        dictionary.filter_extremes()
        corpus = [dictionary.doc2bow(doc) for doc in self.dataset]
        self.dictionary = dictionary
        self.corpus = corpus

    def _compute_tnd(self):
        """
        takes dataset, tnd parameters, tnd mallet path, and computes tnd model on dataset
        sets self.tnd_noise_distribution to the noise distribution computed in tnd

        :return: void
        """
        model = TNDMallet(self.mallet_tnd_path, self.corpus, num_topics=self.tnd_k, id2word=self.dictionary,
                          workers=self.tnd_workers,
                          alpha=self.tnd_alpha, beta=self.tnd_beta0, skew=self.tnd_beta1,
                          iterations=self.tnd_iterations, noise_words_max=self.tnd_noise_words_max,
                          random_seed=self.random_seed)
        noise = model.load_noise_dist()
        self.tnd_noise_distribution = noise

    def _compute_lda(self):
        """
        takes dataset, lda parameters, lda mallet path, and computes LDA model on dataset
        sets self.lda_tw_dist to the topic word distribution computed in LDA

        :return: void
        """
        model = LDAMallet(self.mallet_lda_path, self.corpus, num_topics=self.lda_k, id2word=self.dictionary,
                          workers=self.lda_workers, iterations=self.lda_iterations, random_seed=self.random_seed)
        topic_word_distribution = model.load_word_topics()
        self.lda_tw_dist = topic_word_distribution
        self.lda_model = model
        topics = model.show_topics(num_topics=self.lda_k, num_words=self.nlda_topic_depth, formatted=False)
        self.lda_topics = [[w for (w, _) in topics[i][1]] for i in range(0, len(topics))]

    def _compute_nlda(self):
        """
        takes self.tnd_noise_distribution, self.lda_tw_dist, self.phi, self.top_words, and computes NLDA topics
        sets self.topics to the set of topics computed from noise distribution and topic word distribution

        :return: void
        """
        topics = []
        for i in range(0, len(self.lda_topics)):
            topic = self.lda_topics[i]
            final_topic = []
            j = 0
            while len(topic) > j and len(final_topic) < self.top_words and j < self.nlda_topic_depth:
                w = topic[j]
                token_id = self.dictionary.token2id[w]
                beta = 2
                if w in self.tnd_noise_distribution:
                    beta += self.tnd_noise_distribution[w]
                beta = max(2, beta * (self.nlda_phi / self.lda_k))
                alpha = 2 + self.lda_tw_dist[i, token_id]
                roll = random.betavariate(alpha=math.sqrt(alpha), beta=math.sqrt(beta))
                if roll >= 0.5:
                    final_topic.append(w)
                    if w not in self.tnd_noise_distribution:
                        self.tnd_noise_distribution[w] = 0
                    self.tnd_noise_distribution[w] += (alpha - 2)
                j += 1
            topics.append(final_topic)
        self.topics = topics

    def get_topics(self, top_words=None):
        """
        takes top_words and self.topics, returns a list of topic lists of length top_words

        :param top_words: number of words per topic
        :return: list of topic lists
        """
        if top_words is None:
            top_words = self.top_words
        topics = self.topics
        if topics is None or len(topics) < 1:
            raise ValueError('No topics have been computed yet.')

        return [x[:top_words] for x in topics]
    
    def get_coherent(self):
        coherence_model = CoherenceModel(model = self.lda_model, texts=self.dataset, dictionary=self.dictionary, coherence='u_mass')
        return coherence_model.get_coherence()
    
    def show_topics(self):
        return self.lda_model.show_topics(num_words=self.top_words, formatted=False)
    
    #def assign_topic(self):
    
    
    def get_noise_distribution(self, tnd_noise_words_max=None):
        """
        takes self.tnd_noise_distribution and tnd_noise_words_max
        returns a list of (noise word, frequency) tuples ranked by frequency

        :param tnd_noise_words_max: number of words to be returned
        :return: list of (noise word, frequency) tuples
        """
        if tnd_noise_words_max is None:
            tnd_noise_words_max = self.tnd_noise_words_max
        noise = self.tnd_noise_distribution
        if noise is None or len(noise) < 1:
            raise ValueError('No noise distribution has been computed yet.')

        noise_list = sorted([(x, int(noise[x])) for x in noise.keys()], key=lambda x: x[1], reverse=True)
        return noise_list[:tnd_noise_words_max]
