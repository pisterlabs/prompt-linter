# ##import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Gensim
from gensim.models import CoherenceModel
from gensim.models.ldamulticore import LdaMulticore

#lda vis
import pyLDAvis
import pyLDAvis.gensim_models
import pyLDAvis.gensim_models as gensimvis

import os
from pathlib import Path
import sys
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import pickle

""" Set Mode """
mode = int(sys.argv[1])
opt_nr = int(sys.argv[2])

""" Import Data """
text_types = ["posts", "comments", "all"]
text_type = text_types[mode-1]

print("Load corpora")
#lemma
with open('../data/auxiliary/lemma_' + text_type + '.pkl',  'rb') as f:
    lemma = pickle.load(f)

#corpus
with open('../data/auxiliary/corpus_' + text_type + '.pkl',  'rb') as f:
    corpus = pickle.load(f)

#dicts
with open('../data/auxiliary/dict_' + text_type + '.pkl',  'rb') as f:
    id2word = pickle.load(f)

""" Model Evaluation/Comparison """
Path("../data/images").mkdir(exist_ok=True, parents=True)
Path("../data/results/topics").mkdir(exist_ok=True, parents=True)

if not os.path.isfile("../data/results/topics/ldamodel_" + text_type + "_" + str(opt_nr) + ".pkl"):
    print("Fit Model")
    if mode == 1:
        passes = 5
        iterations = 250
    else:
        passes = 2
        iterations = 50

    ldamodel=LdaMulticore(corpus=corpus, id2word=id2word, num_topics=opt_nr, alpha="asymmetric", chunksize=10000, random_state=0, iterations=iterations, passes=passes, workers=4)
    coherencemodel = CoherenceModel(model=ldamodel, texts=lemma, coherence='c_npmi')
    coherence_score = coherencemodel.get_coherence()

    with open("../data/results/topics/coherence_score_" + text_type + "_" + str(opt_nr) + ".pkl", "wb") as f:
        pickle.dump(coherence_score, f)

    with open("../data/results/topics/ldamodel_" + text_type + "_" + str(opt_nr) + ".pkl", "wb") as f:
        pickle.dump(ldamodel, f)
else:
    with open("../data/results/topics/coherence_score_" + text_type + "_" + str(opt_nr) + ".pkl", "rb") as f:
        coherence_score = pickle.load(f)

print("Coherence score: " + str(round(coherence_score, 3)))


