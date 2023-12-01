# Run in python console
from turtle import pd
import nltk; nltk.download('stopwords')
import re
import numpy as np
import pandas as pd
from pprint import pprint

import json
import codecs

import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import Coherencemodel

import os
from gensim.models import LdaModel

import spacy

import pyLDAvis
import pyLDAvis.gensim_models
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['information', 'geographic', 'geospatial'])

