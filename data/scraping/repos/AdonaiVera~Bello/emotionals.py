import pandas as pd
import numpy as np
import re
import itertools
import streamlit as st
from collections import Counter
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('ggplot')
sns.set_context("talk")
from scipy import stats
from pprint import pprint #Manipulacion de datos
import datetime
import dateutil
from streamlit import components
#LDA MODEL FOR OBSERVACIONES
#quitar mas profundamente stop_words
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
import unicodedata
import tqdm
import spacy.cli
from spacy.lang.es.stop_words import STOP_WORDS
#descargamos los modelos
from nltk.corpus import stopwords
nltk.download('stopwords')
spacy.cli.download("es_core_news_md")
from sklearn.feature_extraction.text import CountVectorizer
import sys
import plotly.express as px
import matplotlib.pyplot as plt
from collections import defaultdict
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import preprocessor as p
import plotly.express as px
import os
from sklearn.preprocessing import MinMaxScaler
sys.setrecursionlimit(10000)
#############LSTM
import sklearn
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
#clasificacion de sentimientos, en español
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from classifier import *
clf = SentimentClassifier()
from pylab import rcParams
## APLLY LDA MODEL TO OBSERVACIONES
#Gensim para modelado de temas, indexación de documentos y recuperación de similitudes con grandes corpus
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
#Spacy para la lemmatization
import spacy
# Herramientas de graficado
import pyLDAvis
import pyLDAvis.gensim
# Habilitado de logging para gensim (opcional)
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.colors as mcolors



class emotionals:
   
    def emotion_count(text,vocab):
        # Separamos las palábras por espacios.
        words=text.split(" ")
        # Creamos un diccionario donde se guardarán los conteos por cada emoción.
        counts={i:0 for i in list(vocab.keys())}
        # Creamos un diccionario donde se guardarán las palabras coincidentes con cada léxico.
        words_per_emo={i:[] for i in list(vocab.keys())}
        # Iteramos para cada una de las palábras dentro del texto.
        for word in words:
            # Iteramos para cada una de las emociones del léxico.
            for emo in vocab:
                # Evalúamos si la palabra está dentro del léxico de cada emoción
                if word in vocab[emo]:
                    # Si la palabra está en el léxico de la emoción, sumamos 1 al conteo acumulado.
                    counts[emo]+=1
                    # También agregamos la palabra coincidente.
                    words_per_emo[emo].append(word)
        return counts, words_per_emo
