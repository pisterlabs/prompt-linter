from gensim.corpora.dictionary import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from gensim.models import KeyedVectors
import gensim.downloader as api
import abc
import numpy as np
import itertools
import torch
from sentence_transformers import util
from scipy.spatial.distance import cosine, jensenshannon
from scipy.stats import entropy

from evaluation.rbo import rbo

# ----- JSD between two sets of topic distributions -----
class JSDivergence(abc.ABC):
    def __init__(self, topic_distrib1, topic_distrib2):
        """
         :param doc_distribution_original_language: numpy array of the topical distribution of
         the documents in the original language (dim: num docs x num topics)
         :param doc_distribution_unseen_language: numpy array of the topical distribution of the
          documents in an unseen language (dim: num docs x num topics)
         """
        super().__init__()
        self.topics1 = topic_distrib1
        self.topics2 = topic_distrib2
        if self.topics1.shape[0] != self.topics2.shape[0]:
            raise Exception('Distributions of the comparable documents must have the same length')

    def score(self):
        """
        :return: average Jensen-Shannon Divergence between the distributions
        """
        jsd2 = compute_jsd(self.topics1.T, self.topics2.T)
        jsd2[jsd2 == np.inf] = 0
        mean_div2 = np.mean(jsd2)

        return mean_div2


def compute_jsd(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
    p /= p.sum()
    q /= q.sum()
    m = (p + q) / 2
    return (entropy(p, m) + entropy(q, m)) / 2


# ----- Text-image matching -----
class TextImageMatching(abc.ABC):
    """
    :param doc_topics: matrix/tensor of L x D x K which contains the doc-topic distributions for docs in the test set where
    L: no. of languages (assume L=2 for now)
    D: no. fo docs
    K: no. of topics
    """
    def __init__(self, doc_topics, image_topics, titles):
        self.doc_topics = doc_topics
        self.image_topics = image_topics
        self.titles = titles

    # compute MRR of first relevant image
    def mrr_first_score(self):
        # get unique article titles
        unique_titles = list(set(self.titles))
        mrr_collect = []
        for title in unique_titles:
            # get all image indices for this title
            true_indices = np.where(np.array(self.titles) == title)[0]
            doc_index = true_indices[0]
            # get JSD between this article and all images in the test data
            doc_dist = np.repeat([self.doc_topics[doc_index]], repeats=self.image_topics.shape[0], axis=0)
            scores = compute_jsd(doc_dist, self.image_topics)
            # rank images according to lowest JSD
            pred_indices = np.argsort(scores)
            for rank, index in enumerate(pred_indices):
                # found first relevant image
                if index in true_indices:
                    mrr_first = 1/(rank+1)
                    mrr_collect.append(mrr_first)
                    break
        mrr_collect = np.mean(mrr_collect)
        return mrr_collect


    # compute UAP
    def uap_score(self):
        # get unique article titles
        unique_titles = list(set(self.titles))
        print("unique articles:", len(unique_titles))
        uap_collect = []
        for title in unique_titles:
            # get all image indices for this title
            true_indices = np.where(np.array(self.titles) == title)[0]
            index_text = true_indices[0]
            # get JSD between this article and all images in the test data
            query_doc_dist = np.repeat([self.doc_topics[index_text]], repeats=self.image_topics.shape[0], axis=0)
            scores = compute_jsd(query_doc_dist.T, self.image_topics.T)
            # rank images according to lowest JSD
            pred_indices = np.argsort(scores)
            # find all positions where a relevant image is found and compute precision
            prec_values = []
            for rank, index in enumerate(pred_indices):
                if index in true_indices:
                    prec = (len(prec_values)+1)/(rank+1)
                    prec_values.append(prec)
            mean_prec = np.mean(prec_values)
            uap_collect.append(mean_prec)
        uap = np.mean(uap_collect)
        return uap

# ----- Cross-lingual Document Retrieval -----
class CrosslingualRetrieval(abc.ABC):
    """
    :param doc_topics: matrix/tensor of L x D x K which contains the doc-topic distributions for docs in the test set where
    L: no. of languages (assume L=2 for now)
    D: no. fo docs
    K: no. of topics
    """
    def __init__(self, doc_topics1, doc_topics2):
        self.doc_topics1 = doc_topics1
        self.doc_topics2 = doc_topics2

    # MRR to evaluate document retrieval performance
    def mrr_score(self):
        total_docs = self.doc_topics1.shape[0]
        total_MRR = 0
        for doc_index in range(total_docs):
            # get JSD between query doc in lang1 and candidate docs in lang2
            query_doc_distrib = np.repeat([self.doc_topics1[doc_index]], repeats=self.doc_topics2.shape[0], axis=0)
            scores = compute_jsd(query_doc_distrib.T, self.doc_topics2.T)
            # compute the MRR
            pred_indices = np.argsort(scores)
            matching_index = np.where((pred_indices == doc_index) == True)[0][0]
            # indices are zero-indexed but MRR assumes top position is index-1 so we add 1 to every index
            MRR = float(1/(matching_index+1))
            total_MRR += MRR
        final_MRR = total_MRR / total_docs
        return final_MRR


# ----- Original -----
class Measure:
    def __init__(self):
        pass

    def score(self):
        pass


class TopicDiversity(Measure):
    def __init__(self, topics):
        super().__init__()
        self.topics = topics

    def score(self, topk=25):
        """
        :param topk: topk words on which the topic diversity will be computed
        :return:
        """
        if topk > len(self.topics[0]):
            raise Exception('Words in topics are less than topk')
        else:
            unique_words = set()
            for t in self.topics:
                unique_words = unique_words.union(set(t[:topk]))
            td = len(unique_words) / (topk * len(self.topics))
            return td


class Coherence(abc.ABC):
    """
    :param topics: a list of lists of the top-k words
    :param texts: (list of lists of strings) represents the corpus on which the empirical frequencies of words are computed
    """
    def __init__(self, topics, texts):
        self.topics = topics
        self.texts = texts
        self.dictionary = Dictionary(self.texts)

    @abc.abstractmethod
    def score(self):
        pass


class CoherenceNPMI(Coherence):
    def __init__(self, topics, texts):
        super().__init__(topics, texts)

    def score(self, topk=10):
        """
        :param topk: how many most likely words to consider in the evaluation
        :return: NPMI coherence
        """
        if topk > len(self.topics[0]):
            raise Exception('Words in topics are less than topk')
        else:
            npmi = CoherenceModel(topics=self.topics, texts=self.texts, dictionary=self.dictionary,
                                  coherence='c_npmi', topn=topk)
            return npmi.get_coherence()


class CoherenceUMASS(Coherence):
    def __init__(self, topics, texts):
        super().__init__(topics, texts)

    def score(self, topk=10):
        """
        :param topk: how many most likely words to consider in the evaluation
        :return: UMass coherence
        """
        if topk > len(self.topics[0]):
            raise Exception('Words in topics are less than topk')
        else:
            umass = CoherenceModel(topics=self.topics, texts=self.texts, dictionary=self.dictionary,
                                   coherence='u_mass', topn=topk)
            return umass.get_coherence()


class CoherenceUCI(Coherence):
    def __init__(self, topics, texts):
        super().__init__(topics, texts)

    def score(self, topk=10):
        """
        :param topk: how many most likely words to consider in the evaluation
        :return: UCI coherence
        """
        if topk > len(self.topics[0]):
            raise Exception('Words in topics are less than topk')
        else:
            uci = CoherenceModel(topics=self.topics, texts=self.texts, dictionary=self.dictionary,
                                 coherence='c_uci', topn=topk)
            return uci.get_coherence()


class CoherenceCV(Coherence):
    def __init__(self, topics, texts):
        super().__init__(topics, texts)

    def score(self, topk=10):
        """
        :param topk: how many most likely words to consider in the evaluation
        :return: C_V coherence
        """
        if topk > len(self.topics[0]):
            raise Exception('Words in topics are less than topk')
        else:
            cv = CoherenceModel(topics=self.topics, texts=self.texts, dictionary=self.dictionary,
                                   coherence='c_v', topn=topk)
            return cv.get_coherence()


class CoherenceWordEmbeddings(Measure):
    def __init__(self, topics, word2vec_path=None, binary=False):
        """
        :param topics: a list of lists of the top-n most likely words
        :param word2vec_path: if word2vec_file is specified, it retrieves the word embeddings file (in word2vec format) to
         compute similarities between words, otherwise 'word2vec-google-news-300' is downloaded
        :param binary: if the word2vec file is binary
        """
        super().__init__()
        self.topics = topics
        self.binary = binary
        if word2vec_path is None:
            self.wv = api.load('word2vec-google-news-300')
        else:
            self.wv = KeyedVectors.load_word2vec_format(word2vec_path, binary=binary)

    def score(self, topk=10, binary= False):
        """
        :param topk: how many most likely words to consider in the evaluation
        :return: topic coherence computed on the word embeddings similarities
        """
        if topk > len(self.topics[0]):
            raise Exception('Words in topics are less than topk')
        else:
            arrays = []
            for index, topic in enumerate(self.topics):
                if len(topic) > 0:
                    local_simi = []
                    for word1, word2 in itertools.combinations(topic[0:topk], 2):
                        if word1 in self.wv.vocab and word2 in self.wv.vocab:
                            local_simi.append(self.wv.similarity(word1, word2))
                    arrays.append(np.mean(local_simi))
            return np.mean(arrays)


class InvertedRBO(Measure):
    def __init__(self, topics):
        """
        :param topics: a list of lists of words
        """
        super().__init__()
        self.topics = topics

    def score(self, topk = 10, weight=0.9):
        """
        :param weight: p (float), default 1.0: Weight of each agreement at depth d:
        p**(d-1). When set to 1.0, there is no weight, the rbo returns to average overlap.
        :return: rank_biased_overlap over the topics
        """
        if topk > len(self.topics[0]):
            raise Exception('Words in topics are less than topk')
        else:
            collect = []
            for list1, list2 in itertools.combinations(self.topics, 2):
                rbo_val = rbo.rbo(list1[:topk], list2[:topk], p=weight)[2]
                collect.append(rbo_val)
            return 1 - np.mean(collect)


class Matches(Measure):
    def __init__(self, doc_distribution_original_language, doc_distribution_unseen_language):
        """
         :param doc_distribution_original_language: numpy array of the topical distribution of
         the documents in the original language (dim: num docs x num topics)
         :param doc_distribution_unseen_language: numpy array of the topical distribution of the
          documents in an unseen language (dim: num docs x num topics)
         """
        super().__init__()
        self.orig_lang_docs = doc_distribution_original_language
        self.unseen_lang_docs = doc_distribution_unseen_language
        if len(self.orig_lang_docs) != len(self.unseen_lang_docs):
            raise Exception('Distributions of the comparable documents must have the same length')

    def score(self):
        """
        :return: proportion of matches between the predicted topic in the original language and
        the predicted topic in the unseen language of the document distributions
        """
        matches = 0
        for d1, d2 in zip(self.orig_lang_docs, self.unseen_lang_docs):
            if np.argmax(d1) == np.argmax(d2):
                matches = matches + 1
        return matches/len(self.unseen_lang_docs)


class KLDivergence(Measure):
    def __init__(self, doc_distribution_original_language, doc_distribution_unseen_language):
        """
         :param doc_distribution_original_language: numpy array of the topical distribution of
         the documents in the original language (dim: num docs x num topics)
         :param doc_distribution_unseen_language: numpy array of the topical distribution of the
          documents in an unseen language (dim: num docs x num topics)
         """
        super().__init__()
        self.orig_lang_docs = doc_distribution_original_language
        self.unseen_lang_docs = doc_distribution_unseen_language
        if len(self.orig_lang_docs) != len(self.unseen_lang_docs):
            raise Exception('Distributions of the comparable documents must have the same length')

    def score(self):
        """
        :return: average kullback leibler divergence between the distributions
        """
        kl_mean = 0
        for d1, d2 in zip(self.orig_lang_docs, self.unseen_lang_docs):
            kl_mean = kl_mean + kl_div(d1, d2)
        return kl_mean/len(self.unseen_lang_docs)


def kl_div(a, b):
    a = np.asarray(a, dtype=np.float)
    b = np.asarray(b, dtype=np.float)
    return np.sum(np.where(a != 0, a * np.log(a / b), 0))


class CentroidDistance(Measure):
    def __init__(self, doc_distribution_original_language, doc_distribution_unseen_language, topics, word2vec_path=None,
                 binary=True, topk=10):
        """
         :param doc_distribution_original_language: numpy array of the topical distribution of the
         documents in the original language (dim: num docs x num topics)
         :param doc_distribution_unseen_language: numpy array of the topical distribution of the
         documents in an unseen language (dim: num docs x num topics)
         :param topics: a list of lists of the top-n most likely words
         :param word2vec_path: if word2vec_file is specified, it retrieves the word embeddings
         file (in word2vec format) to compute similarities between words, otherwise
         'word2vec-google-news-300' is downloaded
         :param binary: if the word2vec file is binary
         :param topk: max number of topical words
         """
        super().__init__()
        self.topics = [t[:topk] for t in topics]
        self.orig_lang_docs = doc_distribution_original_language
        self.unseen_lang_docs = doc_distribution_unseen_language
        if len(self.orig_lang_docs) != len(self.unseen_lang_docs):
            raise Exception('Distributions of the comparable documents must have the same length')

        if word2vec_path is None:
            self.wv = api.load('word2vec-google-news-300')
        else:
            self.wv = KeyedVectors.load_word2vec_format(word2vec_path, binary=binary)

    def score(self):
        """
        :return: average centroid distance between the words of the most likely topic of the
        document distributions
        """
        cd = 0
        for d1, d2 in zip(self.orig_lang_docs, self.unseen_lang_docs):
            top_words_orig = self.topics[np.argmax(d1)]
            top_words_unseen = self.topics[np.argmax(d2)]

            centroid_lang = self.get_centroid(top_words_orig)
            centroid_en = self.get_centroid(top_words_unseen)

            cd += (1 - cosine(centroid_lang, centroid_en))
        return cd/len(self.unseen_lang_docs)

    def get_centroid(self, word_list):
        vector_list = []
        for word in word_list:
            if word in self.wv.vocab:
                vector_list.append(self.wv.get_vector(word))
        vec = sum(vector_list)
        return vec / np.linalg.norm(vec)

