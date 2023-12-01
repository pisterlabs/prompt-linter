import pandas as pd
import numpy as np
import gensim
import nltk
import logging
import pickle

from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from gensim.models.coherencemodel import CoherenceModel
from gensim import corpora, models
from gensim.test.utils import datapath

from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
from scipy.stats import entropy
import sys
sys.path.append('src')
from libs.my_progress_bar import MyBar


np.random.seed(2020)

nltk.download('wordnet')
stemmer = SnowballStemmer('english')

# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
base_path = "D:/work/stackoverflow"
base_model = base_path + "/models_data/"
base_dataset = base_path + "/dataset/"

processed_docs = pd.Series()
for i in range(1, 2):
    processed_docs = processed_docs.append(pd.read_pickle(
        f"{base_dataset}/proc_docs{i}.ser"), ignore_index=True)

dictionary = gensim.corpora.Dictionary(processed_docs)
dictionary.filter_extremes(no_above=0.4, keep_n=300000)
# dictionary.filter_extremes(no_below=30, no_above=0.6, keep_n=300000)

bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

tfidf = models.TfidfModel(bow_corpus)
corpus_tfidf = tfidf[bow_corpus]
print(dictionary)
print(type(bow_corpus))
print(type(tfidf))
print(type(corpus_tfidf))

lda_model_tfidf = gensim.models.LdaModel.load(
    datapath(f"{base_model}lda/model_100n"))

bar = MyBar(max=len(lda_model_tfidf[corpus_tfidf]))

doc_topic_dist = np.zeros(
    (len(lda_model_tfidf[corpus_tfidf]), lda_model_tfidf.num_topics))

for row, doc in enumerate(lda_model_tfidf[corpus_tfidf]):
    for word in doc:
        doc_topic_dist[row][word[0]] = word[1]
    bar.next()
bar.finish()

to_save = doc_topic_dist.tolist()
pickle.dump(f"{base_model}doc_topic_dist")
