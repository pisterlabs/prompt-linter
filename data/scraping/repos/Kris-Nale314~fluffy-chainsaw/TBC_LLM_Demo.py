# Utilities
import time, psutil, os
from os import path
from PIL import Image
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Data manipulation
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_theme()
import json


# Plotting and visualization
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)

# NLP
import string, re, nltk
from string import punctuation
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from num2words import num2words
from spellchecker import SpellChecker
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
#from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import spacy
import spacy_transformers
from spacy import displacy
#from spacy download en_core_web_sm
from transformers import pipeline

from spacy.cli import download
spacy.cli.download("en_core_web_trf")
import en_core_web_trf
#nlp = en_core_web_trf.load()
nlp = spacy.load('en_core_web_trf')


# Scipy
import scipy
from scipy import sparse
from scipy.sparse import csr_matrix

# Mahchine Learning
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split, ParameterGrid

from sklearn.decomposition import NMF
from xgboost import XGBClassifier
#tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
#model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
sentiment_pipeline = pipeline("sentiment-analysis", model = "ProsusAI/finbert")

# Others
import PyPDF2
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from google.colab import drive
import openai

#LDA Visulaization
#from __future__ import print_function
import pyLDAvis
#import pyLDAvis.sklearn
pyLDAvis.enable_notebook()

#start timer
start = time.time()

temp_all= " "

