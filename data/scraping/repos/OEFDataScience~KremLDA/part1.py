#Modules
import pandas as pd
from pprint import pprint
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import gensim.test.utils 
import spacy
import en_core_web_sm
import nltk 
#nltk.download('stopwords') - if needed
from nltk.corpus import stopwords
import tqdm
#Data
df = pd.read_csv("putin_scrape.csv")
