#standard packages
import pandas as pd
import numpy as np
import re
from pprint import pprint

#webscraping packages & SQLite
import pymongo
from pymongo import MongoClient
from bs4 import BeautifulSoup, SoupStrainer
import requests
import urllib.request
import functions as mf

#nlp packages
import string
import nltk
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
from gensim.test.utils import common_texts
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
nlp = spacy.load('en_core_web_lg')

#EDA packages
from collections import Counter, OrderedDict
import matplotlib.pyplot as plt
import seaborn as sns

from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors 
from gensim.scripts.glove2word2vec import glove2word2vec
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import seaborn as sns
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib

# Import the 3 dimensionality reduction methods
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
client = MongoClient()
db = client.case_files
cases = db.cases


class gather_data():
    
    def __init__(self, token, starting_url):
        self.token = token
        self.starting_url = starting_url
        
    
    @staticmethod
    def into_mongo(json):
        for i in json['results']:
            _dict = {'case_id' : i['id'], 'frontend_url': i['frontend_url'], 'case_name': i['name'], 
                      'decision_date': i['decision_date'],
                      'court_name': i['court']['name'], 
                      'court_id': i['court']['id'], 
                      'judges': i['casebody']['data']['judges'],
                      'attorneys': i['casebody']['data']['attorneys'],
                      'case_text': i['casebody']['data']['opinions']}

            x = cases.insert_one(_dict)
      
               
    def scrape(self):
        r = requests.get(f'{self.starting_url}', headers={'Authorization': f'Token {self.token}'})
        #r.status_code
        temp = r.json()
        self.into_mongo(temp)
        next_url = temp['next']
        
        
        while next_url != None:
            
            r = requests.get(f'{next_url}', headers={'Authorization': f'Token f{self.token}'})
            print(r.status_code)
            temp = r.json()
            self.into_mongo(temp)
            next_url = temp['next']
            print(next_url)
            
            
            
class clean_data():
    
    @staticmethod
    def quick_clean(case_text_entry):
        return (case_text_entry[0]['text'].lower().translate(str.maketrans('', '', string.punctuation)).replace('\n', ' ').replace('•', '').replace('“', '').replace('”', ''))

    @staticmethod
    def combine_other_opinions(case_text_entry):
        return [case_text_entry[i]['text'].lower().translate(str.maketrans('', '', string.punctuation)).replace('\n', ' ').replace('•', '') for i in range(1, len(case_text_entry))]
    
    #@staticmethod
    def stop_word_remove(tokenized_text):
        token_list = []
        for token in tokenized_text:
            token_list.append(token.text)
            filtered_sentence =[] 
        for word in token_list:
            lexeme = nlp.vocab[word]
            if lexeme.is_stop == False:
                    filtered_sentence.append(word) 
        return filtered_sentence
    
    
    def clean_data(df):
        df['attorneys'] = df['attorneys'].apply(lambda x: None if len(x) == 0  else x[0].split(','))
        df['judges'] = df['judges'].apply(lambda x: None if len(x) == 0 else x[0])
        df['majority_opinion'] = df['case_text'].apply(lambda x: None if len(x) == 0 else clean_data.quick_clean(x))
        df['other_opinions'] = df['case_text'].apply(lambda x: None if len(x) == 0 else clean_data.combine_other_opinions(x))
#         df['other_opinions'] = df['other_opinions'].apply(lambda x: None if len(x) == 0 else (x[0].lower()))
        df['tokenized_majority_opinion'] = df['majority_opinion'].apply(lambda x: nlp.tokenizer(x) if x != None else False)
#         df['tokenized_other_opinions'] = df['other_opinions'].apply(lambda x: nlp.tokenizer(x) if x != None else False)
        df['no_stop_words_majority'] = df['tokenized_majority_opinion'].apply(lambda x: clean_data.stop_word_remove(x) if x != False else False)
#         df['no_stop_words_other'] = df['tokenized_other_opinions'].apply(lambda x: clean_data.stop_word_remove(x) if x != False else False)                                                      
        return df
        
        
        
class visualizations():
   
    
    @staticmethod
    def sent_to_words(sentences):
        for sentence in sentences:
            yield(gensim.utils.simple_preprocess(str(sentence), deacc=True)) 
         
    
    def bar_plot(list_of_words):    
        counter = Counter(list_of_words)
        counter_top = OrderedDict(counter.most_common(50))
        words = counter_top.keys()
        word_counts = counter_top.values()

        plt.figure(figsize=(10,6))
        indexes = np.arange(len(words))
        width = .7

        plt.bar(indexes, word_counts, width, color = 'lightblue')
        plt.xticks(indexes + width, words, rotation='vertical')
        plt.title('top 50 words across all documents & frequency')
        plt.show()


        
       
        
        
        
        
        