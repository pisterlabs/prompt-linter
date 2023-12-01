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

# Load Data
with open('corp.pkl', 'rb') as f:
    data_lemmatized, _, _ = pickle.load(f)

# Initialize Parameters
count = 0
total_time = 0
coherence_arr = []
time_arr = []

# Set Data State to that of existing model in simulation
data = data_lemmatized[:INITIAL_DOC_SIZE]
id2word = corpora.HashDictionary(data)
# WARNING: Make sure that the dictionary and corpus are always in sync.
# If the mappings in the dictionary and the ids in the corpus are different you will get KeyErrors.
corpus = [id2word.doc2bow(doc) for doc in data]

# Building for the first time - To be considered as the starting/existing model in simulation.
start = timeit.default_timer()
lda = LdaMulticore(corpus, num_topics=35, id2word=id2word,
                   workers=3, chunksize=2000, passes=10, batch=True)
end = timeit.default_timer()

time_taken = end - start
total_time += time_taken
time_arr.append(time_taken)

with open('models_batch.pkl', 'ab') as f2:
    pickle.dump(
        (count, lda, time_arr, data, id2word, len(data)), f2)

# The loop simulates arrival of new documents from Google Alerts in batches of STEP_SIZE
for i in range(INITIAL_DOC_SIZE, len(data_lemmatized)-STEP_SIZE, STEP_SIZE):
    # new_docs is the list of STEP_SIZE new documents which have arrived
    new_docs = data_lemmatized[i:i+STEP_SIZE]

    # Updating Dictionary
    id2word.add_documents(new_docs)
    id2word.filter_extremes(no_below=5, no_above=0.95,
                            keep_n=1800)

    # Converting Documents to doc2bow format so that they can be fed to models
    corpus = [id2word.doc2bow(doc) for doc in data_lemmatized[:i+STEP_SIZE]]
    count += 1
    print('MODEL NO:'+str(count))
    start = timeit.default_timer()
    # Since this is batch LDA the entire model has to be retrained from scratch using the new corpus and dictionary
    lda = LdaMulticore(corpus, num_topics=35, id2word=id2word,
                       workers=3, chunksize=2000, passes=10, batch=True)
    end = timeit.default_timer()

    time_taken = end - start
    total_time += time_taken
    time_arr.append(time_taken)

    # Coherence of model updated so far on the entire corpus from start i.e including Mass Media DB
    # coherencemodel2 = CoherenceModel(
    #     model=lda, texts=data_lemmatized[:i+STEP_SIZE], dictionary=id2word, coherence='c_v')
    # coherence_arr.append(coherencemodel2.get_coherence())

    with open('models_batch.pkl', 'ab') as f2:
        pickle.dump(
            (count, lda, time_arr, data_lemmatized[:i+STEP_SIZE], id2word, i), f2)

print(total_time)
