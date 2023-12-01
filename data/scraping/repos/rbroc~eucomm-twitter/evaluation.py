from gensim.corpora.dictionary import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from gensim.models import KeyedVectors
import gensim.downloader as api
from scipy.spatial.distance import cosine
import abc
import numpy as np
import itertools


class Measure:
    def __init__(self):
        pass

    def score(self):
        pass

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
        self.wv = api.load('glove-twitter-25')
        
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
                        if word1 in self.wv.index_to_key and word2 in self.wv.index_to_key:
                            local_simi.append(self.wv.similarity(word1, word2))
                    arrays.append(np.mean(local_simi))
            return np.mean(arrays)