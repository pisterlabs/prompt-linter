import numpy as np
import pandas as pd

import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.test.utils import datapath

import spacy
from spacy.lemmatizer import Lemmatizer
from spacy.lang.en.stop_words import STOP_WORDS

from tqdm import tqdm as tqdm
from pprint import pprint

lda = gensim.models.LdaModel.load('saved_models/classification_model/gensim_lda_model.gensim')

 # Create a new corpus, made of previously unseen documents.
other_texts = [
    ['computer', 'time', 'graph'],
    ['survey', 'response', 'eps'],
    ['human', 'system', 'computer']]
other_corpus = [common_dictionary.doc2bow(text) for text in other_texts]
unseen_doc = other_corpus[0]
vector = lda[unseen_doc]  # get topic probability distribution for a document
pprint(vector)
