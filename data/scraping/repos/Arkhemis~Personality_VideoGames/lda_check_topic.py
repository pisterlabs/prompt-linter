def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn #Just to ignore some annoying deprecation warnings
import os
import random
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import pandas as pd
from gensim import corpora
import numpy as np
import matplotlib.pyplot as plt
import gensim
import spacy
import warnings
from pprint import pprint
from gensim.models.ldamodel import LdaModel
from gensim.models.wrappers import LdaMallet

nlp_model = spacy.load('en_core_web_sm')
nlp_model.max_length = 1000000000
SEED = 1962 #For reproducing the results
random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)



df = pd.read_csv('./df_review_complete.csv')
df = df.dropna(subset=["tokens"], inplace=True).reset_index()

data_words = np.array(df['tokens'])
del df #Clear some memory
bigram = gensim.models.Phrases(data_words, min_count=10)
trigram = gensim.models.Phrases(bigram[data_words])
for idx in range(len(data_words)):
    for token in bigram[data_words[idx]]:
        if '_' in token:
            # Token is a bigram, add to document.
            data_words[idx].append(token)
    for token in trigram[data_words[idx]]:
        if '_' in token:
            # Token is a bigram, add to document.
            data_words[idx].append(token)
data_words = [d.split() for d in data_words]
id2word = corpora.Dictionary(data_words)
id2word.filter_extremes(no_below=10, no_above=0.2) #Filtering frequent words
corpus = [id2word.doc2bow(text) for text in data_words]


from gensim.models.wrappers import LdaMallet
os.environ.update({'MALLET_HOME':r'./mallet'})
mallet_path = './mallet/bin/mallet' # update this path


def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    coherence_values = []
    for num_topics in range(start, limit, step):
        model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=id2word, random_seed = SEED)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_val = coherencemodel.get_coherence()
        coherence_values.append(coherence_val)
        print("num_topics = ",num_topics, "has a coherence of:", coherence_val)

    return coherence_values
model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=data_words, start=50, limit=100, step=5)
