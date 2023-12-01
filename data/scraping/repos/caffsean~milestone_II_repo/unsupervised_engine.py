# Import modules
import numpy as np
import pandas as pd
from gensim.models import LsiModel
import re, nltk, spacy, gensim
from gensim import corpora
from gensim.models import LsiModel
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from gensim.models.coherencemodel import CoherenceModel
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
import texthero as hero

# Sklearn
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from pprint import pprint

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Plotting tools
#!pip install -U pyLDAvis
import pyLDAvis
import pyLDAvis.sklearn
import matplotlib.pyplot as plt




class unsupervised_engine():
    def choices():
        return ['preprocess_data',
               'prepare_data',
               'create_LSA',
               'get_coherence_values']
    
    def preprocess_data(doc_set):
        tokenizer = RegexpTokenizer(r'\w+')
        en_stop = set(stopwords.words('english'))
        p_stemmer = PorterStemmer()
        texts = []
        for i in tqdm(doc_set):
            raw = i.lower()
            tokens = tokenizer.tokenize(raw)
            stopped_tokens = [i for i in tokens if not i in en_stop]
            stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
            texts.append(stemmed_tokens)
        return texts
    
    def prepare_data(list_text, tfidf = False):
        dictionary = corpora.Dictionary(list_text)
        doc_term_matrix = [dictionary.doc2bow(doc) for doc in list_text] 
        if tfidf == True:
            tmodel = TfidfModel(doc_term_matrix, smartirs='ntc')
            doc_term_matrix = tmodel[doc_term_matrix]
        return dictionary, doc_term_matrix
    
    def create_LSA(list_text, number_topics, num_words, tfidf = False):
        dictionary, doc_term_matrix = unsupervised_engine.prepare_data(list_text)
        lsamodel = LsiModel(doc_term_matrix, num_topics = number_topics, id2word = dictionary)
        return lsamodel
    
    def get_coherence_values(list_text, stop, start = 2, step = 3, lsimodel = True, tfidf = False):
        coherence_values = []
        model_list = []
    
        dictionary, doc_term_matrix = unsupervised_engine.prepare_data(list_text, tfidf = False)
        for num_topics in range(start, stop, step):
        
            if lsimodel == True:
                model = LsiModel(doc_term_matrix, num_topics=num_topics, id2word = dictionary)
            else:
                model = LdaMulticore(doc_term_matrix, num_topics=num_topics, id2word = dictionary)
            model_list.append(model)
        
            coherencemodel = CoherenceModel(model=model, texts=list_text, dictionary=dictionary, coherence='c_v')
            coherence_values.append(coherencemodel.get_coherence())
        return model_list, coherence_values