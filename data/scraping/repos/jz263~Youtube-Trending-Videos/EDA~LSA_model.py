#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import pymc3 as pm
import matplotlib.patches as mpatches


from gensim import corpora
from gensim.models import LsiModel
from nltk.tokenize import RegexpTokenizer,word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt

###sample dataset
class Bayesian_Lsi:
    def __init__(self, n_topics = 5, n_words = 8 ):
        self.n_topics = n_topics
        self.n_words = n_words
        self.dictionary = None
        self.doc_term_matrix = None

    def preprocess_data(self,doc_set):
        """
        Input  : docuemnt list
        Purpose: preprocess text (tokenize, removing stopwords, and stemming)
        Output : preprocessed text
        """
        # initialize regex tokenizer
        tokenizer = RegexpTokenizer(r'\w+')
        # create English stop words list
        en_stop = set(stopwords.words('english'))
        # Create p_stemmer of class PorterStemmer
        p_stemmer = PorterStemmer()
        # list for tokenized documents in loop
        texts = []
        # loop through document list
        for i in doc_set:
            # clean and tokenize document string
            raw = i.lower()
            tokens = tokenizer.tokenize(raw)
            # remove stop words from tokens
            stopped_tokens = [i for i in tokens if not i in en_stop]
            # stem tokens
            stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
            # add tokens to list
            texts.append(stemmed_tokens)
        return texts

    #def prepare_corpus(self,doc_clean):
    #    """
    #    Input  : clean document
    #    Purpose: create term dictionary of our courpus and Converting list of documents (corpus) into Document Term Matrix
    #    Output : term dictionary and Document Term Matrix
    #    """
    #    # Creating the term dictionary of our courpus, where every unique term is assigned an index. dictionary = corpora.Dictionary(doc_clean)
    #    # generate LDA model
    #    return dictionary,doc_term_matrix

    def create_gensim_lsa_model(self,doc_clean):
        """
        Input  : clean document, number of topics and number of words associated with each topic
        Purpose: create LSA model using gensim
        Output : return LSA model
        """
        self.dictionary = corpora.Dictionary(doc_clean)
        # Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
        self.doc_term_matrix = [self.dictionary.doc2bow(doc) for doc in doc_clean]

        #dictionary,doc_term_matrix=self.prepare_corpus(doc_clean)
        # generate LSA model
        lsamodel = LsiModel(self.doc_term_matrix, num_topics=self.n_topics, id2word = self.dictionary)  # train model
        #print(lsamodel.print_topics(num_topics=self.n_topics, num_words=words))
        return lsamodel

    def compute_coherence_values(self, doc_clean, stop, start, step):
        """
        Input   : dictionary : Gensim dictionary
                  corpus : Gensim corpus
                  texts : List of input texts
                  stop : Max num of topics
        purpose : Compute c_v coherence for various number of topics
        Output  : model_list : List of LSA topic models
                  coherence_values : Coherence values corresponding to the LDA model with respective number of topics
        """
        coherence_values = []
        model_list = []
        for num_topics in range(start, stop, step):
            # generate LSA model
            model = LsiModel(self.doc_term_matrix, num_topics=num_topics, id2word = self.dictionary)  # train model
            model_list.append(model)
            coherencemodel = CoherenceModel(model=model, texts=doc_clean, dictionary=self.dictionary, coherence='c_v')
            coherence_values.append(coherencemodel.get_coherence())
        return model_list, coherence_values

    def concat_lsa(self, df_raw, lsamodel):
        '''
        Append the topics probability matrix of each document into training dataset.
        The
        '''
        # the shape of df_raw and lsa_corpus must match,df_raw must be reindexed
        # convert corpus to dataframe for the further merge with model dataset
        lsa_corpus = lsamodel[self.doc_term_matrix]
        lsa_array = np.ndarray(shape = (len(lsa_corpus),self.n_topics))
        for i in range(len(lsa_corpus)): ###return a line of one document
            for j in range(len(lsa_corpus[i])):   #return a tuple of word with index in one doc
                lsa_array[i][j] = lsa_corpus[i][j][1]

        #create column names
        Columns =np.char.add('topic_', np.linspace(0,self.n_topics-1,self.n_topics,dtype = int).astype('str'))

        ###convert to dataframe
        lsa_df = pd.DataFrame(data = lsa_array,columns = Columns)
        ### standarize probs
        lsa_df = (lsa_df -lsa_df.mean())/lsa_df.std()
        ### concatenate lsa with raw dataset
        df_concat_w_lsa =pd.concat([df_raw,lsa_df],axis = 1,)
        return df_concat_w_lsa

    def plot_graph(self, doc_clean,start, stop, step):
        dictionary = corpora.Dictionary(doc_clean)
        # Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
        doc_term_matrix = [self.dictionary.doc2bow(doc) for doc in doc_clean]
        model_list, coherence_values = Bayesian_Lsi.compute_coherence_values(dictionary, doc_term_matrix,doc_clean,
                                                                stop, start, step)
        #store max coherence score
        Opt_number_of_topics = np.argmax(coherence_values)+start

        # Show graph
        x = range(start, stop, step)
        f, (ax2) = plt.subplots(1, 1, figsize = (8,6))
        #plt.axis([0, 10, 0, 10])
        plt.plot(x, coherence_values)
        plt.xlabel("Number of Topics")
        plt.ylabel("Coherence score")
        plt.legend(("coherence_values"), loc='best')
        Opt = mpatches.Patch(color='#aaff80', label="Opt_number_of_topics = {}".format(Opt_number_of_topics))
        ax2.legend(handles=[Opt],loc = 'upper left')
        plt.show()
