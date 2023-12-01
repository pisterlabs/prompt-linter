import pandas as pd
import nltk
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import networkx as nx
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import os.path
from gensim import corpora
from gensim.models import LsiModel
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from gensim.models.coherencemodel import CoherenceModel


def find_optimal_clusters(data, max_k):
    iters = range(1, max_k+1, 2)
    
    sse = []
    for k in iters:
        sse.append(MiniBatchKMeans(n_clusters=k, init_size=1024, batch_size=2048, random_state=20).fit(data).inertia_)
        print('Fit {} clusters'.format(k))
        
    f, ax = plt.subplots(1, 1)
    ax.plot(iters, sse, marker='o')
    ax.set_xlabel('Cluster Centers')
    ax.set_xticks(iters)
    ax.set_xticklabels(iters)
    ax.set_ylabel('SSE')
    ax.set_title('SSE by Cluster Center Plot')
    
find_optimal_clusters(text, 15)

df=pd.read_csv("web.csv", sep=";", encoding='utf8')

df["DE"]=df["DE"].str.lower()

df.loc[((df["DE"].str.contains("digital twin")) | (df["DE"].str.contains("digital twin;"))| (df["DE"].str.contains("digital factory")) | (df["DE"].str.contains("digital thread"))| (df["DE"].str.contains("digital shadow"))| (df["DE"].str.contains("smart factory"))| (df["DE"].str.contains("smart manufacturing"))),"manter"]=1

df.loc[df["manter"]==1].to_csv("manter.csv", sep=";")

df2 = df.loc[df["manter"]==1]

tfidf = TfidfVectorizer(
    min_df = 5,
    max_df = 0.95,
    max_features = 8000,
    stop_words = 'english'
)
tfidf.fit(df2.AB)
text = tfidf.transform(df2.AB)

plot_tsne_pca(text, clusters)

