# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 08:52:17 2018

@author: chuang
"""

### Guided LDA 

from gensim import corpora, models 
import numpy as np
import sys
import os
import gensim
import pickle
import nltk
from collections import Counter
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
from gensim.models import CoherenceModel
import matplotlib
import matplotlib.pyplot as plt
#%matplotlib inline

python_root = './scripts'
sys.path.insert(0, python_root)

