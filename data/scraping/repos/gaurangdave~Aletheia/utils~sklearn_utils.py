import numpy as np
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


# Styling
def color_green(val):
    color = 'green' if val > .1 else 'black'
    return 'color: {col}'.format(col=color)


def make_bold(val):
    weight = 700 if val > .1 else 400
    return 'font-weight: {weight}'.format(weight=weight)


def print_sklearn_sparcity(data_vectorized):
    # Materialize the sparse data
    data_dense = data_vectorized.todense()

    # Compute Sparsicity = Percentage of Non-Zero cells
    print("Sparsicity: ", ((data_dense > 0).sum()/data_dense.size)*100, "%")


def create_sklearn_dominent_topic_dataframe(lda_model, data_vectorized):
    lda_output = lda_model.transform(data_vectorized)
    # column names
    topicnames = ["Topic" + str(i) for i in range(lda_model.n_components)]
    # index names
    docnames = ["Doc" + str(i) for i in range(len(data))]

    df_document_topic = pd.DataFrame(
        np.round(lda_output, 2), columns=topicnames, index=docnames)
    # Get dominant topic for each document
    dominant_topic = np.argmax(df_document_topic.values, axis=1)
    df_document_topic['dominant_topic'] = dominant_topic
    return df_document_topic


def print_sklearn_dominant_topics(lda_model, data_vectorized):
    df_document_topic = create_sklearn_dominent_topic_dataframe(
        lda_model, data_vectorized)
    # Apply Style
    df_document_topics = df_document_topic.head(
        15).style.applymap(color_green).applymap(make_bold)
    return df_document_topics


def print_sklearn_topic_distribution(lda_model, data_vectorized):
    df_document_topic = create_sklearn_dominent_topic_dataframe(
        lda_model, data_vectorized)
    df_topic_distribution = df_document_topic['dominant_topic'].value_counts(
    ).reset_index(name="Num Documents").rename(columns={'index': 'Topic'})
    # df_topic_distribution.columns = ["Topic Num", "Num Documents"]
    return df_topic_distribution


# Show top n keywords for each topic
def show_sklearn_topics(vectorizer, lda_model, n_words=20):
    keywords = np.array(vectorizer.get_feature_names_out())
    topic_keywords = []
    for topic_weights in lda_model.components_:
        top_keyword_locs = (-topic_weights).argsort()[:n_words]
        topic_keywords.append(keywords.take(top_keyword_locs))
    return topic_keywords


def format_sklearn_topics(topic_keywords):
    df_topic_keywords = pd.DataFrame(topic_keywords)
    df_topic_keywords.columns = [
        'Word '+str(i) for i in range(df_topic_keywords.shape[1])]
    df_topic_keywords.index = [
        'Topic '+str(i) for i in range(df_topic_keywords.shape[0])]
    return df_topic_keywords


def analyze_sklearn_lda_model(lda_model, data_vectorized):
    # Log Likelyhood: Higher the better
    print("Log Likelihood: ", lda_model.score(data_vectorized))
    # Perplexity: Lower the better. Perplexity = exp(-1. * log-likelihood per word)
    print("Perplexity: ", lda_model.perplexity(data_vectorized))

# helper function to visualize lda model


def visualize_sklearn_lda_model(lda_model, data_vectorized, vectorizer, mds='tsne'):
    pyLDAvis.enable_notebook()
    panel2 = pyLDAvis.sklearn.prepare(
        lda_model, data_vectorized, vectorizer, mds=mds)
    return panel2
