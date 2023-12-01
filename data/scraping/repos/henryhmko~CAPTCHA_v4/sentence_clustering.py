import cohere 
import nltk
from nltk import tokenize
import random
from collections import defaultdict
import pickle
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist, pdist
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pickle

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')

co = cohere.Client('INSERT_TOKEN_HERE') 



text = open("Book 1 - The Philosopher's Stone.txt").read()
text = text.replace('”', ".")
text = text.replace('“', ".")
text = tokenize.sent_tokenize(text)
text = [s for s in text if len(tokenize.word_tokenize(s)) > 15]
#len(text)

response = co.embed( 
  model='large', 
  texts=text) 

data = response.embeddings




kmeans = KMeans(n_clusters=6)
label = kmeans.fit_predict(data)

with open('sentences', 'wb') as fp:
    pickle.dump(text, fp)

with open('labels', 'wb') as fp:
    pickle.dump(label, fp)
    