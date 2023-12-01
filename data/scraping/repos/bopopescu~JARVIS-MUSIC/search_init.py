import pandas as pd
import numpy as np
from collections import Counter
import re

# languange processing imports
#import nltk
import string
from gensim.corpora import Dictionary
# preprocessing imports
from sklearn.preprocessing import LabelEncoder
from pprint import pprint
# model imports
from gensim.utils import simple_preprocess

from gensim.models import CoherenceModel, Phrases, phrases
from gensim.models.doc2vec import LabeledSentence
from gensim.models.doc2vec import TaggedDocument
from gensim.models.doc2vec import Doc2Vec
from gensim.test.utils import get_tmpfile
from sklearn.preprocessing import normalize

#import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import scipy
import warnings
import os
warnings.filterwarnings("ignore",category=FutureWarning)
warnings.filterwarnings("ignore",category=DeprecationWarning)


# read lyrics.csv
def process_corpus(column):
    lyric_dir = os.path.join('lyricsMatch', 'semantic_search', 'lyrics.csv')
    data = pd.read_csv(lyric_dir, names=['singer', 'song', 'lyrics'])
    data = data[~data['lyrics'].isna()]

    # data_words_nostops = remove_stopwords(data.lyrics)
    # data_lemmatized = lemmatization(data_words_nostops, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    # corpus = list(map(lambda words: " ".join(words), data_lemmatized))

    count_vect = CountVectorizer().fit(data[column].values)
    words_count = count_vect.transform(data[column].values)
    tf_transformer = TfidfTransformer(sublinear_tf=True).fit(words_count)
    words_tfidf = tf_transformer.transform(words_count)
    return data, count_vect, tf_transformer, words_tfidf
    # scipy.sparse.save_npz('sparse_matrix.npz',words_tfidf)
    # words_tfidf = scipy.sparse.load_npz('sparse_matrix.npz')

data, count_vect_lyric, tf_transformer_lyric, words_tfidf_lyric = process_corpus('lyrics')
data, count_vect_name, tf_transformer_name, words_tfidf_name = process_corpus('song')


model_file = os.path.join(os.getcwd(), 'lyricsMatch', 'semantic_search', "my_doc2vec_model")
fname = get_tmpfile(model_file)
model = Doc2Vec.load(fname)
