from tqdm.auto import tqdm
from joblib import dump, load
import pickle
import plotly.io as io
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis.sklearn
import pyLDAvis
from gensim.models.ldamulticore import LdaMulticore
from gensim.models.ldamodel import LdaModel
from gensim.models import CoherenceModel
from gensim.models.phrases import Phraser
from gensim.models import Phrases
from gensim.utils import simple_preprocess
from gensim import corpora, models
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import spacy
from pprint import pprint
import pandas as pd

import re
import string
from bs4 import BeautifulSoup
import nltk
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('stopwords')


# Plotting tools


# loading library


# function to compute optimal parameters for LDA model

def compute_coherence_values(corpus, id2word, texts, num_topics, passes, chunk_sizes=[200], iterations=[100]):
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
    params = []
    for num_topic in tqdm(num_topics):
        # for chunk_size in tqdm(chunk_sizes):
        for num_passes in tqdm(passes):
            for iteration in tqdm(iterations):
                model = LdaModel(corpus=corpus,
                                 id2word=id2word,
                                 num_topics=num_topic,
                                 random_state=100,
                                 update_every=1,
                                 # chunksize=chunk_size,
                                 passes=num_passes,
                                 iterations=iteration,
                                 per_word_topics=True)
                model_list.append(model)
                # Compute Perplexity
                perplexity = model.log_perplexity(corpus)

                # Compute Coherence Score
                coherence_model_lda = CoherenceModel(
                    model=model, texts=texts, dictionary=id2word, coherence='c_v')
                cv_coherence = coherence_model_lda.get_coherence()

                coherence_model_umass = CoherenceModel(
                    model=model, texts=texts, dictionary=id2word, coherence='u_mass')
                umass_coherence = coherence_model_umass.get_coherence()

                coherence_values.append({
                    "perplexity": perplexity,
                    "cv_coherence": cv_coherence,
                    "umass_coherence": umass_coherence,
                })
                params.append({'num_topics': num_topic, 'chunk_size': "chunk_size",
                              'passes': num_passes, 'iterations': iteration})

    return model_list, coherence_values, params


def analyze_gensim_lda_model(lda_model, corpus, id2word, texts, num_topics, passes, chunk_sizes=[200]):
    # Compute Perplexity
    # a measure of how good the model is. lower the better.
    print('\nPerplexity: ', lda_model.log_perplexity(corpus))
    # Compute Coherence Score
    coherence_model_lda = CoherenceModel(
        model=lda_model, texts=texts, dictionary=id2word, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score: ', coherence_lda)

# helper functions to visualize LDA model


def visualize_gensim_lda_model(lda_model, corpus, id2word, filename="gensim_lda.html"):
    # Visualize the topics
    pyLDAvis.enable_notebook()
    vis = gensimvis.prepare(lda_model, corpus, id2word)
    vis.save(filename)
