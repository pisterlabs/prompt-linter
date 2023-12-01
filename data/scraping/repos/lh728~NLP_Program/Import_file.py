import sys
import re, numpy as np, pandas as pd
from pprint import pprint
import gensim, spacy, logging, warnings # gensim,spacy package needs pip install
import gensim.corpora as corpora
import seaborn as sns
import matplotlib.colors as mcolors 
'''
Gensim previously only wrapped the lemmatization routines of another library (Pattern) - 
this was not a particularly modern/maintained option, so removed from Gensim-4.0
from gensim.utils import lemmatize
so we choose from nltk.stem import WordNetLemmatizer
'''
from nltk.stem import WordNetLemmatizer
from gensim.utils import simple_preprocess
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import matplotlib.colors as mcolors
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt
plt.style.use('seaborn')
from nltk.corpus import stopwords
%matplotlib inline
warnings.filterwarnings("ignore",category=DeprecationWarning)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)
from sklearn.datasets import fetch_20newsgroups
import nltk
nltk.download('stopwords')
import json
from spacy.lang.en import English
from collections import Counter
from sklearn.manifold import TSNE
from bokeh.plotting import figure, output_file, show
from bokeh.models import Label
import pyLDAvis.gensim_models
