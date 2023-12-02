"""
Class for training latent semantic models
"""
#import pandas as pd
import numpy as np
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel

class LatentSemantics:
    def __init__(self, texts, titles=False, k=2, mallet_path="/home/knielbo/Mallet/bin/mallet"):

        self.texts = texts

        if titles:
            self.titles = titles
        else:
            self.titles = ["text_{}".format(i) for i in range(len(texts))]
        
        self.mallet = mallet_path
        self.k = k

    def generate_id2word(self):
        return corpora.Dictionary(self.texts)

    def generate_corpus(self):
        id2word = self.generate_id2word()
        return [id2word.doc2bow(text) for text in self.texts]
    
    def fit(self):
        self.id2word = self.generate_id2word()
        self.corpus = self.generate_corpus()
        self.model = gensim.models.wrappers.LdaMallet(self.mallet,
                                                 corpus=self.corpus,
                                                 num_topics=self.k,
                                                 id2word=self.id2word,
                                                 workers=8,
                                                 optimize_interval=5,
                                                 random_seed=41
                                                 )
        self.coherencemodel = CoherenceModel(model=self.model, texts=self.texts, dictionary=self.id2word, coherence="c_v")
        self.coherence = self.coherencemodel.get_coherence()
    
    def coherence_k(self, krange=[10,20,30,40,50], texts=False):
        k_cohers = list()
        for (i, k) in enumerate(krange):
            print("[INFO] Estimating coherence model for k = {}, iteration {}".format(k, i))
            ls = LatentSemantics(self.texts, k=k)
            ls.fit()
            #k_cohers.append((k, ls.coherence))
            k_cohers.append(ls.coherence)


        k_cohers = np.array(k_cohers, dtype=np.float)
        idx =  k_cohers.argsort()[-len(krange):][::-1]
        k = krange[idx[np.argmax(k_cohers[idx]) & (np.gradient(k_cohers)[idx] >= 0)][0]]
        
        return k, k_cohers