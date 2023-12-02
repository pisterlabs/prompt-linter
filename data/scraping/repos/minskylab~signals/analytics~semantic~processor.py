import pandas as pd
import numpy as np
import os
import environs
import os.path
from gensim import corpora
from gensim.models import LsiModel
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from gensim.models.coherencemodel import CoherenceModel

from db import DB, Query, Tweet

nltk_stopwords_fetched = False

class SemanticProcessor:
    def __init__(self, db_instance: DB):
        global nltk_stopwords_fetched
        self.db = db_instance
        self.banned_words = ["http", "https", "com", "twitter", "fund", "relief", "pic", "www", "pe"]
        if not nltk_stopwords_fetched:
            nltk.download('stopwords')
            nltk_stopwords_fetched = True

    def retrieve_data(self, query=Query):
        df = run_query_to_df(db_instance=self.db, query=query)
        self.tweets = df
        return self

    def set_data(self, tweets: pd.DataFrame):
        self.tweets = tweets
        return self
        
    def preprocess(self):
        documents_list = [str(t) for t in self.tweets["tweet"]]
        tokenizer = RegexpTokenizer(r'\w+')
        es_stop = set(stopwords.words('spanish'))
        en_stop = set(stopwords.words('english'))
        p_stemmer = PorterStemmer()
        texts = []
        for i in documents_list:
            raw = i.lower()
            tokens = tokenizer.tokenize(raw)
            stopped_tokens = [i for i in tokens if not i in es_stop]
            stopped_tokens = [i for i in stopped_tokens if not i in en_stop]
            stopped_tokens = [i for i in stopped_tokens if not i in self.banned_words]
            stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
            texts.append(stemmed_tokens)
            
        self.texts = texts
        # corpus
        self.dictionary = corpora.Dictionary(self.texts)
        self.doc_term_matrix = [self.dictionary.doc2bow(doc) for doc in self.texts]
        return self

    def measure_coherence(self):
        # finding better coherence
        start = 2
        step = 1
        stop = 12
        model_coherence_values = []

        for num_topics in range(start, stop, step):
            # generate LSA model
            model = LsiModel(self.doc_term_matrix, num_topics=num_topics, id2word = self.dictionary)  # train model
            coherencemodel = CoherenceModel(model=model, texts=self.texts, dictionary=self.dictionary, coherence='c_v')
            model_coherence_values.append((coherencemodel.get_coherence(), num_topics, model))

        _, best_num_topics, best_model = max(model_coherence_values, key = lambda i : i[0])
        self.best_num_topics = best_num_topics
        self.model = best_model
        return self

    def retrieve_topics(self, number_of_words=6):
        topics_showed = self.model.show_topics(num_topics=self.best_num_topics, num_words=number_of_words, formatted=False)
        return topics_showed

