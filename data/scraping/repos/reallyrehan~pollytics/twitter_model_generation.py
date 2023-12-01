import sys
if 'google.colab' in sys.modules:
    !pip install emoji --upgrade
    !pip install pandas-profiling==2.*
    !pip install plotly==4.*
    #!python -m spacy download en_core_web_lg
    !pip install pyldavis
    !pip install gensim
    !pip install chart_studio
    !pip install --upgrade autopep8
#Base and Cleaning 
import json
import requests
import pandas as pd
import numpy as np
import emoji
import regex
import re
import string
from collections import Counter

#Visualizations
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt 
import pyLDAvis.gensim
import chart_studio
import chart_studio.plotly as py 
import chart_studio.tools as tls

#Natural Language Processing (NLP)
import spacy
import gensim
from spacy.tokenizer import Tokenizer
from gensim.corpora import Dictionary
from gensim.models.ldamulticore import LdaMulticore
from gensim.models.coherencemodel import CoherenceModel
from gensim.parsing.preprocessing import STOPWORDS as SW
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from pprint import pprint
from wordcloud import STOPWORDS
from gensim import corpora
import pickle
import gensim
import re
from nltk.corpus import wordnet as wn
import spacy
from nltk.stem.wordnet import WordNetLemmatizer
from spacy.lang.en import English
import random
from datetime import datetime


def tokenize(text):
    lda_tokens = []
    tokens = parser(text)
    for token in tokens:
        if token.orth_.isspace():
            continue
        elif token.like_url:
            url_list.append(token)
        elif token.orth_.startswith('@'):
            user_list.append(token)
        else:
            lda_tokens.append(token.lower_)
    return lda_tokens
def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma

def get_lemma2(word):
    return WordNetLemmatizer().lemmatize(word)

def prepare_text_for_lda(text):
    hashtag_list.append(re.findall(r"#(\w+)", text))
    tokens = tokenize(text)
    tokens = [token for token in tokens if len(token) > 4]
    tokens = [token for token in tokens if token not in en_stop]
    tokens = [get_lemma(token) for token in tokens]
    tokens = [token for token in tokens if token != "SCREEN_NAME"]

    return tokens

def prepare_text_for_lda(text):
    hashtag_list.append(re.findall(r"#(\w+)", text))
    tokens = tokenize(text)
    tokens = [token for token in tokens if len(token) > 4]
    tokens = [token for token in tokens if token not in en_stop]
    tokens = [get_lemma(token) for token in tokens]
    tokens = [token for token in tokens if token != "SCREEN_NAME"]

    return tokens

def give_emoji_free_text(text):
    """
    Removes emoji's from tweets
    Accepts:
        Text (tweets)
    Returns:
        Text (emoji free tweets)
    """
    emoji_list = [c for c in text if c in emoji.UNICODE_EMOJI]
    clean_text = ' '.join([str for str in text.split() if not any(i in str for i in emoji_list)])
    return clean_text

def url_free_text(text):
    '''
    Cleans text from urls
    '''
    text = re.sub(r'http\S+', '', text)
    return text

stopwords = set(STOPWORDS)
nltk.download('wordnet')
nltk.download('stopwords')
en_stop = set(nltk.corpus.stopwords.words('english'))
from spacy.lang.en import English
spacy.load('en')
parser = English()



st_date_object = datetime.strptime("2020-01-01", '%Y-%m-%d')

b_tweet = []
d_tweet = []
for bd in biden_tweets:
  if st_date_object<=datetime.strptime(bd['UTC'].split('T')[0], '%Y-%m-%d'):
    b_tweet.append(bd['Text'])
    d_tweet.append(datetime.strptime(bd['UTC'].split('T')[0], '%Y-%m-%d'))

biden_df = pd.DataFrame({"text":b_tweet,"date":d_tweet})

df = biden_df
call_emoji_free = lambda x: give_emoji_free_text(x)

df['emoji_free_tweets'] = df['text'].apply(call_emoji_free)

df['url_free_tweets'] = df['emoji_free_tweets'].apply(url_free_text)


url_list = []
user_list = []
hashtag_list = []




tokens = []

for doc in df['url_free_tweets']:
    doc_tokens = []    
    doc_tokens = prepare_text_for_lda(doc)
    tokens.append(doc_tokens)

# Makes tokens column
df['tokens'] = tokens

id2word = Dictionary(df['tokens'])

id2word.filter_extremes(no_below=2, no_above=.99)

corpus = [id2word.doc2bow(d) for d in df['tokens']]


# Instantiating a Base LDA model 
base_model = LdaMulticore(corpus=corpus, num_topics=10, id2word=id2word, workers=12, passes=5)

words = [re.findall(r'"([^"]*)"',t[1]) for t in base_model.print_topics()]

topics = [' '.join(t[0:10]) for t in words]

# Getting the topics
for id, t in enumerate(topics): 
    print(f"------ Topic {id} ------")
    print(t, end="\n\n")

p=pyLDAvis.gensim.prepare(base_model, corpus, id2word)
pyLDAvis.save_html(p, 'biden_lda.html')

ldamodel.save('biden_model.gensim')

biden_df=df



st_date_object = datetime.strptime("2020-01-01", '%Y-%m-%d')

b_tweet = []
d_tweet = []
for bd in trump_tweets:
  if st_date_object<=datetime.strptime(bd['date'].split(' ')[0], '%Y-%m-%d'):
    b_tweet.append(bd['text'])
    d_tweet.append(datetime.strptime(bd['date'].split(' ')[0], '%Y-%m-%d'))

trump_df = pd.DataFrame({"text":b_tweet,"date":d_tweet})
df = trump_df


call_emoji_free = lambda x: give_emoji_free_text(x)

df['emoji_free_tweets'] = df['text'].apply(call_emoji_free)

df['url_free_tweets'] = df['emoji_free_tweets'].apply(url_free_text)


url_list = []
user_list = []
hashtag_list = []


tokens = []

for doc in df['url_free_tweets']:
    doc_tokens = []    
    doc_tokens = prepare_text_for_lda(doc)
    tokens.append(doc_tokens)

# Makes tokens column
df['tokens'] = tokens

id2word = Dictionary(df['tokens'])

id2word.filter_extremes(no_below=2, no_above=.99)

corpus = [id2word.doc2bow(d) for d in df['tokens']]

base_model = LdaMulticore(corpus=corpus, num_topics=10, id2word=id2word, workers=12, passes=5)

words = [re.findall(r'"([^"]*)"',t[1]) for t in base_model.print_topics()]

topics = [' '.join(t[0:10]) for t in words]

# Getting the topics
for id, t in enumerate(topics): 
    print(f"------ Topic {id} ------")
    print(t, end="\n\n")

p=pyLDAvis.gensim.prepare(base_model, corpus, id2word)
pyLDAvis.save_html(p, 'trump_lda.html')

ldamodel.save('trump_model.gensim')

trump_df=df
