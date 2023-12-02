import os,pickle,random 
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from gensim import corpora, models
from gensim.models.phrases import Phrases, Phraser
import gensim
from sklearn.metrics import fbeta_score
from sklearn.metrics import f1_score
from gensim.models import CoherenceModel
from scipy.stats import entropy
import logging

class Model(object):
    def __init__(self):
        self.path='../data/files/'
        self.df_pkl=pickle.load(open(self.path+'df_final.pkl','rb'))
    
    def get_EC_list(self,df):
        EC_list_init=df['EC'].values.tolist()
        EC_list=[list(set(i)) for i in EC_list_init]
        return EC_list

    def get_train_and_test(self,df):
        for i in range(0,10000):
            train,test=train_test_split(df,test_size=0.075)
            test_crop=test[['Map','Name','EC_all_cleaned']]
            test_crop.rename(columns={'EC_all_cleaned':'EC'}, inplace=True)
            train_crop=train[['Map','Name','EC_all_cleaned']]
            train_crop.rename(columns={'EC_all_cleaned':'EC'}, inplace=True)
            EC_unique_train=set([j for i in train_crop['EC'].to_list() for j in i])
            EC_unique_test=set([j for i in test_crop['EC'].to_list() for j in i])
            if EC_unique_test.issubset(EC_unique_train):
                print ("True")
            break
        EC_list_train=self.get_EC_list(train_crop)
        EC_list_test=self.get_EC_list(test_crop)
        return EC_list_train,EC_list_test,train_crop,test_crop

    def get_train_and_test_for_FT(self,df):
        train,test=train_test_split(df,test_size=0.15)
        test_crop=test[['Map','Name','EC_all_cleaned']]
        test_crop.rename(columns={'EC_all_cleaned':'EC'}, inplace=True)
        train_crop=train[['Map','Name','EC_all_cleaned']]
        train_crop.rename(columns={'EC_all_cleaned':'EC'}, inplace=True)
        EC_list_train=self.get_EC_list(train_crop)
        EC_list_test=self.get_EC_list(test_crop)
        return EC_list_train,EC_list_test,train_crop,test_crop

    def get_dict_corpus(self,train):
        dictionary = gensim.corpora.Dictionary(train)
        corpus = [dictionary.doc2bow(EC) for EC in train]
        return dictionary,corpus

    def get_test_corpus(self,dictionary,test):
        test_corpus=[dictionary.doc2bow(EC) for EC in test]
        return test_corpus

    def MyLDA(self,corpus,dictionary,num_topics,random_state,passes):
        logging.basicConfig(filename='models/lda_model.log', format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        ldamodel=gensim.models.ldamodel.LdaModel(corpus, num_topics=num_topics, 
                                      random_state=random_state,
                                      id2word = dictionary, passes=passes)
        #ldamodel.save('models/lda_train_150topics_100passes.model')
        return ldamodel

    def MyLDAP(self,corpus,dictionary,num_topics,alpha,eta,random_state,passes):
        ldamodel=gensim.models.ldamodel.LdaModel(corpus, num_topics=num_topics,
                                      random_state=random_state,
                                      id2word = dictionary, 
                                      alpha=alpha,
                                      eta=eta,
                                      passes=passes)
        return ldamodel
    

    def model_perplexity(self,model,corpus):
        perplexity=model.log_perplexity(corpus)
        return perplexity

    def model_coherence(self,model,texts,dictionary):
        coherence_model_lda = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        return coherence_lda

if __name__=='__main__':
    EC_list_train,EC_list_test=Model().get_train_and_test(Model().df_pkl)
    dictionary,corpus=Model().get_dict_corpus(EC_list_train)
    ldamodel=Model().MyLDA(corpus,dictionary,num_topics=150,random_state=250,passes=100)
    perplexity=Model().model_perplexity(ldamodel,corpus)
    test_corpus=Model().get_test_corpus(dictionary,EC_list_test)
    perplexity_test=Model().model_perplexity(ldamodel,test_corpus)
    coherence=Model().model_coherence(ldamodel,EC_train=EC_list_train,dictionary=dictionary)
