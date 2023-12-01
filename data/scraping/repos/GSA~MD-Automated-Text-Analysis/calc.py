import re
import numpy as np
import pandas as pd
from pprint import pprint
from collections import OrderedDict

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim import utils, models


# spacy for lemmatization
import spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

import nltk
from nltk.corpus import stopwords
import string
from bs4 import BeautifulSoup
from nltk.stem.wordnet import WordNetLemmatizer

def compute_coherence_values(dictionary, corpus, texts, limit, start, step,id2word):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.ldamodel.LdaModel( corpus=corpus, num_topics=num_topics, id2word=id2word)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values

def print_model(df):
    '''
    creates a model from a text field and graphs the coherence values

    parameters:
    ---------
    dataframe with a 'TEXT' column

    returns:
    ---------
    void
    
    '''
    data_2 = df['TEXT'].values.tolist()

    # Remove Emails
    data_2 = [re.sub('\S*@\S*\s?', '', str(sent)) for sent in data_2]

    # Remove new line characters
    data_2 = [re.sub('\s+', ' ', str(sent)) for sent in data_2]

    # Remove distracting single quotes
    data_2 = [re.sub("\'", "", str(sent)) for sent in data_2]

    data_words_2 = list(sent_to_words(data_2))

    # Build the bigram and trigram models
    bigram2 = gensim.models.Phrases(data_words_2, min_count=5, threshold=50) # higher threshold fewer phrases.
    trigram2 = gensim.models.Phrases(bigram2[data_words_2], threshold=50)  

    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod2 = gensim.models.phrases.Phraser(bigram2)
    trigram_mod2= gensim.models.phrases.Phraser(trigram2)

    # Remove Stop Words
    data_words_nostops2 = remove_stopwords(data_words_2)

    # Form Bigrams
    data_words_bigrams2 = make_bigrams(data_words_nostops2)

    # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
    # python3 -m spacy download en
    nlp = spacy.load('en', disable=['parser', 'ner'])

    # Do lemmatization keeping only noun, adj, vb, adv
    data_lemmatized2 = lemmatization(data_words_bigrams2, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

    # Create Dictionary
    id2word2 = corpora.Dictionary(data_lemmatized2)

    # Create Corpus
    texts2 = data_lemmatized2

    # Term Document Frequency
    corpus2 = [id2word2.doc2bow(text) for text in texts2]

    model_list, coherence_values = compute_coherence_values(dictionary=id2word2, corpus=corpus2, texts=data_lemmatized2, start=2, limit=20, step=2)

    print(coherence_values)
    max_coherence_score = max(coherence_values)
    best_num_loc = coherence_values.index(max_coherence_score)
    best_topic_num = (coherence_values.index(max_coherence_score) + 1) *2
    print (best_topic_num)

    best_model = model_list[best_num_loc]

    model_topics = best_model.show_topics(formatted=False)

    model_2 = gensim.models.ldamodel.LdaModel( corpus=corpus, num_topics=best_topic_num, id2word=id2word)

    #pprint(best_model.print_topics(num_words=8))
    pprint(model_2.print_topics(num_words=8))


    limit=20; start=2; step=2;
    x = range(start, limit, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    plt.show()