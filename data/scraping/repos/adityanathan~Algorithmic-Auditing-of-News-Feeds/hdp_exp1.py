import pandas as pd
import numpy as np
import pickle
import re
import timeit
import spacy

import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel, HdpModel, LdaModel, LdaMulticore
from nltk.corpus import stopwords
import helper as he

# Number of documents to consider as part of existing model in simulation.
# Here it is the no. of documents in Mass Media DB
INITIAL_DOC_SIZE = 17850
STEP_SIZE = 200

# Load Data - corp.pkl contains data_lemmatized, id2word, corpus
with open('corp.pkl', 'rb') as f:
    data_lemmatized, id2word, _ = pickle.load(f)

# Initialize Parameters
count = 0
total_time = 0
coherence_arr = []
time_arr = []

# Set Data State to that of existing model in simulation
data = data_lemmatized[:INITIAL_DOC_SIZE]
# id2word = corpora.HashDictionary(data)
corpus = [id2word.doc2bow(doc) for doc in data]

# Building for the first time - To be considered as the starting/existing model in simulation.
start = timeit.default_timer()
hdpmodel, hdptopics = he.build_hdp(corpus, id2word)
end = timeit.default_timer()

time_taken = end - start
total_time += time_taken
time_arr.append(time_taken)

coherencemodel = CoherenceModel(
    model=hdpmodel, texts=data, dictionary=id2word, coherence='c_v')
coherence_arr.append(coherencemodel.get_coherence())

# The loop simulates arrival of new documents from Google Alerts in batches of STEP_SIZE
for i in range(INITIAL_DOC_SIZE, len(data_lemmatized)-STEP_SIZE, STEP_SIZE):
    # new_docs is the list of STEP_SIZE new documents which have arrived
    new_docs = data_lemmatized[i:i+STEP_SIZE]

    # # Updating Dictionary
    # dict2 = corpora.Dictionary(new_docs)
    # id2word.merge_with(dict2)
    # id2word.filter_extremes(no_below=5, no_above=0.95,
    # keep_n=1800)

    # Converting Documents to doc2bow format so that they can be fed to models
    corpus = [id2word.doc2bow(doc) for doc in new_docs]
    count += 1
    print('MODEL NO:'+str(count))
    start = timeit.default_timer()
    # print(corpus)
    hdpmodel.update(corpus)
    end = timeit.default_timer()

    time_taken = end - start
    total_time += time_taken
    time_arr.append(time_taken)

    coherencemodel = CoherenceModel(
        model=hdpmodel, texts=data_lemmatized[:i+STEP_SIZE], dictionary=id2word, coherence='c_v')
    coherence_arr.append(coherencemodel.get_coherence())

    # Saves the incrementally built up data just in case program fails in the middle.
    # Ideally you want the coherence_arr and time_arr from the last iteration. Other variables have been saved as a precaution.
    with open('models_hdp.pkl', 'ab') as f2:
        pickle.dump(
            (count, hdpmodel, coherence_arr, time_arr, data_lemmatized[:i+STEP_SIZE], id2word, i), f2)

print(total_time)
