
############ Imports ####################
import numpy as np
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess 
from gensim.models.coherencemodel import CoherenceModel 
from gensim.models.ldamodel import LdaModel
import spacy
import nltk
import pyLDAvis 
import warnings
import pickle
import json
from gensim.models import TfidfModel
import matplotlib.pyplot as plt
import re
from pprint import pprint
import pandas as pd
import time
from gensim.test.utils import datapath
from gensim import corpora, models, similarities

warnings.filterwarnings("ignore")

########## LDA Model Building Script #############

# Load in the articles and Id2Words's
df = pd.read_pickle('C:/Users/senic/OneDrive/Desktop/Masters/WINTER_2021/ECO2460/Term_Paper/Code/Replication_Data/cleaned_WSJ_data_V2.pkl')
df_id = pd.read_pickle('C:/Users/senic/OneDrive/Desktop/Masters/WINTER_2021/ECO2460/Term_Paper/Code/Replication_Data/WSJ_id2word_V2.pkl')

corpus = df['corpus'].tolist()
data_bigrams = df['data_bigrams'].tolist()
DOJ = df['text'].tolist()

# now build your final LDA model with your chosen number of topics
tops = 190

# Play with the hyperparameters
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,id2word = df_id,num_topics =tops,
                                            random_state=52,
                                            update_every=5,
                                            chunksize=500,
                                            passes=8,
                                            alpha="auto",
                                            eta = "auto") 


# Save the model
lda_model.save('C:/Users/senic/OneDrive/Desktop/Masters/WINTER_2021/ECO2460/Term_Paper/Code/models/190_model_slow_V5.model')













