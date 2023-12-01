import string
import re
import pandas as pd

import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel


from nltk.corpus import stopwords as sw
from nltk.corpus import wordnet as wn
from nltk import wordpunct_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import sent_tokenize
from nltk import pos_tag

import spacy
nlp = spacy.load('en', disable=['parser', 'ner'])


from sklearn.base import BaseEstimator,TransformerMixin


class NLTKPreprocesor(BaseEstimator,TransformerMixin):
    def __init__(self,stopwords = None,punct = None,lower = True,strip=True):
        self.lower = lower
        self.strip = strip
        self.stopwords = stopwords or set(sw.words('english'))
        self.punct = punct or set(string.punctuation)
        self.lemmatizer = WordNetLemmatizer()

    def fit(self,X,y=None):
        return self

    def inverse_transform(self,X):
        pass

    def transform(self,X):
        return [list(self.tokenize(doc)) for doc in X]

    def tokenize(self,document):
        for sent in sent_tokenize(document):
            for token,tag in pos_tag(wordpunct_tokenize(sent)):
                token = token.lower() if self.lower else token
                token = token.strip() if self.strip else token
                token = token.strip('_') if self.strip else token
                token = token.strip('*') if self.strip else token
                token = token.strip('#') if self.strip else token

                if token in self.stopwords:
                    continue

                if all(char in self.punct for char in token):
                    continue

                if len(token) <= 0:
                    continue

                lemma = self.lemmatize(token,tag)
                yield lemma

    def lemmatize(self,token,tag):
        tag ={
            'N' : wn.NOUN,
            'V' : wn.VERB,
            'R' : wn.ADV,
            'J' : wn.ADJ
        }.get(tag[0],wn.NOUN)

        return self.lemmatizer.lemmatize(token,tag)


def remove_emails(docs):
    return [re.sub('\S*@\S*\s?', '', sent) for sent in docs]


def remove_new_lines(docs):
    return [re.sub('\s+', ' ', sent) for sent in docs]


def sentence_to_words(docs):
    for sent in docs:
        yield(simple_preprocess(str(sent),deacc=True))


def generate_bigram(docs,min_count=10,thresh=100):
    bigram = gensim.models.Phrases(docs,min_count=min_count,threshold=thresh)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    return [bigram_mod[doc] for doc in docs]


def generate_trigram(bigram,docs,min_count=10,thresh=100):
    trigram = gensim.models.Phrases(bigram[docs], threshold=100)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    return [trigram_mod[doc] for doc in docs]


def remove_stopwords(docs):
    stopwords = set(sw.words('english'))
    return [[word for word in simple_preprocess(str(doc)) if word not in stopwords] for doc in docs]


def lemmatization(docs, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in docs:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


def clean_up_data(raw_data:pd.DataFrame):
    data = raw_data.values.tolist()
    print('READING DATA...',flush=True)
    data = remove_emails(data)
    print('REMOVING EMAILS...', flush=True)
    data = remove_new_lines(data)
    print('REMOVING NEW LINES...', flush=True)
    data_sents = list(sentence_to_words(data))
    print('GETTING SENTENCES...', flush=True)
    data_sents = remove_stopwords(data_sents)
    print('GETTING BIGRAMS...', flush=True)
    bigrams = generate_bigram(data_sents)
    print('LEMMETIZING BIGRAMS...', flush=True)
    data_lemmetized = lemmatization(bigrams)
    return data_lemmetized

