import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import datetime
import itertools
import pyLDAvis.gensim_models

import string
import spacy

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

nlp = spacy.load('en_core_web_sm')

import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models.phrases import Phrases
from gensim.models import CoherenceModel
from gensim.corpora import Dictionary
from gensim.models import LdaMulticore


from textblob import TextBlob

from collections import defaultdict
from collections import Counter
from datetime import datetime
import math
from operator import itemgetter
import os
import random
import re

from pprint import pprint

from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import matplotlib.colors as mcolors


from gensim.models import LsiModel



words = ['lesbian', 'gay', 'homosexual', 'gender', 'bisexual', 'sexuality', 'same sex',
         'asexual', 'biphobia', 'bisexual', 'coming out', 'coming-out', 'gender identity',
        'queer', 'genderqueer', 'gender-queer', 'homophobia', 'LGBTQ', 'LGBT', 'LGBTQ+', 'LGBTQIA',
        'lgbtq', 'lgbt', 'lgbtq+', 'lgbtqia', 'non binary', 'non-binary', 'transgender', 'come', 'people', 'go', 'coming',
        'think', 'thinking', 'going', 'want', 'wanting', 'wanted', 'know', 'knowing', 'say', 'saying', 'said'] 

stopwords = stopwords.words('english')
spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS
stopwords = list(set(stopwords) | spacy_stopwords)


def remove_bi_and_ally(df):
    """ Since having "bi" and "ally" in the keywords extracted a lot of quotes not related to gay rights, this function
    removes anything with bi and ally that doesn't contain the other keywords
    :param df: the quotes dataframe
    :return: the dataframe with the rows removed
    """
    df = df[~(df.quotation.str.contains("bi") & ~df.quotation.isin(words))]
    df = df[~(df.quotation.str.contains("ally") & ~df.quotation.isin(words))]
    df = df.reset_index(drop=True)
    return df


def remove_keywords(quote):
    """ Removes the keywords from the quotation before clustering
    :param quote: a string containing the quotation
    :return: a list of words not containing the keywords
    """
    split = quote.split()
    final = [word for word in split if word.lower() not in words]
    return final


def sent_to_words(sentences):
    """ Converts a document into a list of lowercase tokens, ignoring tokens that are too short or too long
    :param sentences: the quotation
    :return: list of strings
    """
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))
        
def remove_stopwords(d):
    """ Removes the stopwords
    :param d: the documents
    :return: the documents without stopwords
    """
    return [[word for word in simple_preprocess(str(doc)) if word not in stopwords] for doc in d]

def make_bigrams(d, bigram_mod):
    return [bigram_mod[doc] for doc in d]

def make_trigrams(d):
    return [trigram_mod[bigram_mod[doc]] for doc in d]

def lemmatization(d, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    d_out = []
    for sent in d:
        doc = nlp(" ".join(sent)) 
        d_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return d_out


def nlp_pipe(docs):
    """ Preprocesses the documents and returns the lemmatized data
    :param docs: docs to be preprocessed
    :return: data ready for a model
    """
    
    data = list(sent_to_words(docs))
    
    bigram = gensim.models.Phrases(data, min_count=2, threshold=100)
    trigram = gensim.models.Phrases(bigram[data], threshold=100)  

    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    
    # remove stopwords
    data_nostops = remove_stopwords(data)

    # create bigrams
    data_bigrams = make_bigrams(data_nostops, bigram_mod)

    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

    data_lemmatized = lemmatization(data_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

    return data_lemmatized

def create_corpus(data):
    """ Creates the dictionary and corpus
    :param data: the preprocessed data
    :return: the corpus and dictionary
    """
    id2word = corpora.Dictionary(data)

    max_freq = 0.5
    min_wordcount = 5
    id2word.filter_extremes(no_below=min_wordcount, no_above=max_freq)

    # create corpus
    texts = data

    corpus = [id2word.doc2bow(text) for text in texts]
    
    return corpus, id2word

def run_lda(data_lemmatized, corpus, id2word, num_topics):
    """ Creates and lda model, prints the topics and calculates perplexity and coherence
    :param data_lemmatized: the data
    :param corpus: the corpus
    :param id2word: the dictionary
    :param num_topics: the number of clusters
    :return: perplexity score, coherence score, and the lda model
    """
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=num_topics, 
                                           random_state=100,
                                           update_every=1,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)
    pprint(lda_model.print_topics())
    doc_lda = lda_model[corpus]
    
    perplexity = lda_model.log_perplexity(corpus)
    
    coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
    coherence = coherence_model_lda.get_coherence()
    
    return perplexity, coherence, lda_model




def word_bubble(lda_model, num_topics):
    """ Creates word bubbles for each topic
    :param lda_model: the lda model
    :num_topics: the number of topics
    """
    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

    cloud = WordCloud(stopwords=stopwords,
                      background_color='white',
                      width=2500,
                      height=1800,
                      max_words=10,
                      colormap='Paired',
                      color_func=lambda *args, **kwargs: cols[i],
                      prefer_horizontal=1.0)

    topics = lda_model.show_topics(formatted=False)

    fig, axes = plt.subplots(1, num_topics, figsize=(10,10), sharex=True, sharey=True)

    for i, ax in enumerate(axes.flatten()):
        fig.add_subplot(ax)
        topic_words = dict(topics[i][1])
        cloud.generate_from_frequencies(topic_words, max_font_size=300)
        plt.gca().imshow(cloud)
        plt.gca().axis('off')


    plt.subplots_adjust(wspace=0, hspace=0)
    plt.axis('off')
    plt.margins(x=0, y=0)
    plt.tight_layout()
    plt.show()
    

    


