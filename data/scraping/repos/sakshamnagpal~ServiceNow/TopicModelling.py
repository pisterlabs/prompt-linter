import nltk
import re
import pandas as pd
from pprint import pprint
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt
import warnings
from nltk.corpus import stopwords

warnings.filterwarnings("ignore", category=DeprecationWarning)
nltk.download('stopwords')

stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu'])

