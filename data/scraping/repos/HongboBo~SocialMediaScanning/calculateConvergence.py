# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 09:34:17 2022

@author: nct76
"""

import pickle as pkl
from gensim.models.coherencemodel import CoherenceModel
import numpy as np 

with open('./ldamodels.pkl','rb') as f:
    ldamodels = pkl.load(f)
with open('./texts.pkl','rb') as f:
    texts = pkl.load(f)

allcoh = []
for lda in ldamodels:
    coh = []
    for l in lda:
        cm = CoherenceModel(model=l, texts=texts, coherence='c_v')
        coh.append(cm.get_coherence())
    allcoh.append(coh)
allcoh = np.array(allcoh)


with open('convergence.npy', 'wb') as f:
    np.save(f, allcoh)