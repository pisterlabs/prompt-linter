# Import the necessary libraries
from itertools import combinations
from gensim.models import CoherenceModel
import numpy as np


class CoherenceScores():

    def __init__(self, topic_list, bow, dictionary, w2v):       
        super(CoherenceScores, self).__init__()
        self.topic_list = topic_list
        self.bow = bow
        self.dictionary = dictionary
        self.w2v = w2v
    
    def c_v(self):

        """Calculates the C_V coherence score."""

        model = CoherenceModel(topics=self.topic_list, texts=self.bow, dictionary=self.dictionary, coherence='c_v')
        return model.get_coherence()

    def c_npmi(self):

        """Calculates the NPMI coherence score."""

        model = CoherenceModel(topics=self.topic_list, texts=self.bow, dictionary=self.dictionary, coherence='c_npmi')
        return model.get_coherence()

    def c_uci(self):

        """Calculates the UCI coherence score."""

        model = CoherenceModel(topics=self.topic_list, texts=self.bow, dictionary=self.dictionary, coherence='c_uci')
        return model.get_coherence()

    def c_w2v(self):

        """Calculates the C_W2V coherence score."""

        similarity = []

        for index, topic in enumerate(self.topic_list):
            local_similarity = []
        
            for word1, word2 in combinations(topic, 2):
                if word1 in self.w2v.wv.vocab and word2 in self.w2v.wv.vocab:                    
                    local_similarity.append(self.w2v.wv.similarity(word1, word2))
        
            similarity.append(np.mean(local_similarity))
    
        return np.mean(similarity)


    def get_coherence_scores(self):

        """Returns a list containing the coherence score of C_V, NPMI, UCI, and C_W2V, respectively."""

        C_V = self.c_v()
        NPMI = self.c_npmi()
        UCI = self.c_uci()
        C_W2V = self.c_w2v()
        return [C_V, NPMI, UCI, C_W2V]