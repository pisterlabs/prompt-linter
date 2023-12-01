import os,pickle,random 
import pandas as pd
pd.set_option('mode.chained_assignment', None)
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from gensim import corpora, models
from gensim.models.phrases import Phrases, Phraser
import gensim
from sklearn.metrics import fbeta_score
from sklearn.metrics import f1_score
from gensim.models import CoherenceModel
from scipy.stats import entropy
import logging
import Model

class Parameterization(Model.Model):
    def __init__(self):
        super().__init__()
        self.df=Model.Model().df_pkl
        self.EC_train,self.EC_test,self.train_df,self.test_df=Model.Model().get_train_and_test(Model.Model().df_pkl)
        self.dictionary,self.corpus=Model.Model().get_dict_corpus(self.EC_train)
        self.test_corpus=Model.Model().get_test_corpus(self.dictionary,self.EC_test)

    def MyLDA_param(self,corpus,dictionary,num_topics,alpha,eta,random_state=200,passes=200):
        ldamodel=gensim.models.ldamodel.LdaModel(corpus, num_topics=num_topics,
                                      alpha=alpha,
                                      eta=eta,
                                      random_state=random_state,
                                      id2word = dictionary, passes=passes)
        return ldamodel

    def get_topic_similarity(self,topics,model,distance='jaccard'):
        mdiff, annotation = model.diff(model, distance=distance, num_words=100)
        sim_list=[j for i in mdiff for j in i]
        mean=np.mean(sim_list)
        return mean

    def gridsearch(self,alpha=np.arange(0.01,0.1,0.01),eta=np.arange(0.05,0.5,0.05),topics=np.arange(75,150,5)):
        cols = ['Topics', 'alpha', 'eta','Coherence','log_perplexity','distance_jaccard',
                'distance_JS','distance_hellinger']
        lst = []
        for topic in topics:
            for a in alpha:
                for e in eta:
                    print ("running search for {}, {} ,{}".format(topic,a,e))
                    model=self.MyLDA_param(self.corpus,self.dictionary,topic,a,e)
                    coherence=Model.Model().model_coherence(model=model,texts=self.EC_train,dictionary=self.dictionary)
                    perplexity=Model.Model().model_perplexity(model,corpus=self.test_corpus)
                    distance_JS=self.get_topic_similarity(topics=topic,model=model,distance='jensen_shannon')
                    distance_jaccard=self.get_topic_similarity(topics=topic,model=model,distance='jaccard')
                    distance_hellinger=self.get_topic_similarity(topics=topic,model=model,distance='hellinger')
                    lst.append([topic, a,e,coherence,perplexity,distance_jaccard,
                                distance_JS,distance_hellinger])
                    print([topic, a,e,coherence,perplexity,distance_jaccard,
                                distance_JS,distance_hellinger])
        df = pd.DataFrame(lst, columns=cols)
        df.sort_values(by=['distance_JS'])
        pickle.dump(df, open( "models/gridsearch_re.pkl", "wb" ) )


if __name__=='__main__':
    Parameterization()
    print (Parameterization().df.head())
    Parameterization().gridsearch() #(alpha=[0.1],eta=[0.2],topics=[75])
