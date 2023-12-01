"""
Class for training latent semantic models
"""
import pandas as pd
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel

class LatentSemantics:
    def __init__(self, texts, titles=False, k=5, mallet_path="/home/knielbo/mallet-2.0.8/bin/mallet"):

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
        
        theta_df = pd.read_csv(self.model.fdoctopics(), delimiter='\t', header=None)
        theta_df.drop(theta_df.iloc[:, :2], inplace=True, axis=1)
        self.theta = theta_df.values
        
    #def theta(self)    
        #print(model.print_topic(0,topn=2))
        #self.lda_model = model[1]
        #print(self.lda_model.print_topic(0, topn=2))
        #self.id2word = model[2]
        #self.corpus = model[3]