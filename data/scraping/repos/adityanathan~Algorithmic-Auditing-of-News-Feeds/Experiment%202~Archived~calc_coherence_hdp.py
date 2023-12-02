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

file_name = 'models_hdp.pkl'
output = 'hdp_coherence.pkl'

model_list = []
data_list = []
dict_list = []
with open(file_name, 'rb') as f:
    while True:
        try:
            iteration, model, time_arr, data, id2word, _ = pickle.load(f)
            print(iteration)
            model_list.append(model)
            data_list.append(data)
            dict_list.append(id2word)
        except:
            break

coherence_list = []
count = 0
for i in range(0, len(model_list)):
    model = model_list[i]
    data = data_list[i]
    id2word = dict_list[i]
    print(id2word)
    count += 1
    print('Iteration '+str(count))
    coherencemodel = CoherenceModel(
        model=model, texts=data, dictionary=id2word, coherence='c_v')
    coherence_list.append(coherencemodel.get_coherence())

with open(output, 'wb') as f:
    pickle.dump((coherence_list, time_arr), f)
