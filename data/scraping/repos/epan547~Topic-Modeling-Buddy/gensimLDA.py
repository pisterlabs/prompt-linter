import re
import numpy as np
import pandas as pd
from pprint import pprint
from os import path
import pickle

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt

class LDAModel:
    """Parent class for Mallet and Gensim LDA"""
    def __init__(self, search_term, start, limit, step):
        self.search_term = search_term
        self.id2word, self.corpus, self.texts = self.make_corpus(self.load_cleaned_words())
        self.start = start
        self.limit = limit
        self.step = step
        self.paths = {}    #implement constructor

    def load_cleaned_words(self):
        filename = "./Text/" + self.search_term.replace(" ", "-") + "-words.pckl"
        if path.exists(filename):
            f = open(filename, 'rb')
            tokens = pickle.load(f)
            f.close()
            print(len(tokens))
            print([len(words) for words in tokens])
            return tokens
        else:
            return "this topic has not yet been pickled. run get_data.py first."

    def make_bigrams(self, tokens):
        # Build the bigram and trigram models
        bigram = gensim.models.Phrases(tokens, min_count=5, threshold=10) # higher threshold fewer phrases.
        bigram_mod = gensim.models.phrases.Phraser(bigram)
        return bigram_mod[tokens]

    def make_corpus(self, tokens):
        texts = self.make_bigrams(tokens)
        id2word = corpora.Dictionary(texts)
        corpus = [id2word.doc2bow(doc) for doc in texts]
        return id2word, corpus, texts

    def compute_coherence_values(self):
        """
         Returns:
        -------
        model_list : List of LDA topic models
        coherence_values : Coherence values corresponding to the LDA model
        with respective number of topics
        """
        #should not reach this case in implementation
        print("implement compute_coherence_values in child class")
        return [],[]

    def compute_optimal_num_topics(self):
        model_list, coherence_values = self.compute_coherence_values()  #to be overridden by subclass
        #plot coherence score over num topics
        x = range(self.start, self.limit, self.step)
        plt.plot(x, coherence_values)
        plt.xlabel("Num Topics")
        plt.ylabel("Coherence score")
        plt.legend(("coherence_values"), loc='best')
        plt.show()
        #get model with best coherence score
        max_index = np.argmax(np.asarray(coherence_values))
        return model_list[max_index], coherence_values[max_index]

    def optimize_coherence(self):
        #parameter sweep of different k values
        best_model, max_coherence = self.compute_optimal_num_topics()

        topics = best_model.print_topics()
        pprint(topics)

        # Compute Coherence Score
        print('\nCoherence Score: ', max_coherence)
        return best_model, topics

    def train(self):
        """Do nothing. To be overridden by subclass"""

    def pckl_results(self, path, content):
        f = open(path, 'wb')
        pickle.dump(content, f)
        f.close()
        print("pickled content of type " + str(type(content)) + " at "+ path)

class Mallet(LDAModel):
    def __init__(self, search_term, start, limit, step):
        super().__init__(search_term, start, limit, step)
        self.paths = {'topics':"./Topics/" + search_term.replace(" ", "-") + "-mallet.pckl",
        'mallet-path':'../../mallet-2.0.8/bin/mallet'} #update this path

    def compute_coherence_values(self):
        """
        Compute c_v coherence for various number of topics
        Returns:
        -------
        model_list : List of LDA topic models
        coherence_values : Coherence values corresponding to the LDA model
        with respective number of topics
        """
        coherence_values = []
        model_list = []
        for num_topics in range(self.start, self.limit, self.step):
            #train model with num_topics
            model = gensim.models.wrappers.LdaMallet(self.paths['mallet-path'],
            corpus=self.corpus, num_topics=num_topics, id2word=self.id2word)
            model_list.append(model)

            #calculate coherence
            coherencemodel = CoherenceModel(model=model, texts=self.texts,
            dictionary=self.id2word, coherence='c_v')
            score = coherencemodel.get_coherence()
            print("topic: ", num_topics, "coherence: ", score)
            coherence_values.append(score)

        return model_list, coherence_values

    def train(self):
        best_model, topics = self.optimize_coherence()
        self.pckl_results(self.paths['topics'], topics)

class GensimLDA(LDAModel):
    def __init__(self, search_term, start, limit, step):
        super().__init__(search_term, start, limit, step)
        self.paths = {'topics':"./Topics/" + search_term.replace(" ", "-") + ".pckl",
        'vis':"./LDAVis/" + search_term.replace(" ", "-") + ".pckl"}

    def compute_coherence_values(self):
        """
        Compute c_v coherence for various number of topics
        """
        coherence_values = []
        model_list = []
        for num_topics in range(self.start, self.limit, self.step):
            #train model with num_topics
            model=gensim.models.ldamodel.LdaModel(corpus=self.corpus,
            id2word=self.id2word, num_topics=num_topics)
            model_list.append(model)

            #calculate coherence for model
            coherencemodel = CoherenceModel(model=model, texts=self.texts,
            dictionary=self.id2word, coherence='c_v')
            score = coherencemodel.get_coherence()
            print("topic: ", num_topics, "coherence: ", score)
            coherence_values.append(score)

        return model_list, coherence_values

    def train(self):
        best_model, topics = self.optimize_coherence()
        vis = pyLDAvis.gensim.prepare(best_model, self.corpus, self.id2word)
        self.pckl_results(self.paths['vis'], vis)
        self.pckl_results(self.paths['topics'], topics)

def gensimLDAdriver(search_term, isMallet):
    if isMallet:
        model = Mallet(search_term, 2, 24, 2)
        model.train()
    else:
        model = GensimLDA(search_term, 1, 15, 1)
        model.train()
