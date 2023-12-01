from gensim.test.utils import common_texts
from gensim.corpora import Dictionary
from gensim.models import Word2Vec, WordEmbeddingSimilarityIndex
from gensim.similarities import SoftCosineSimilarity, SparseTermSimilarityMatrix
import pandas as pd 
import numpy as np 
import sklearn as sk 
import csv 
import matplotlib.pyplot as plt
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import re
from pprint import pprint

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy

# Plotting tools
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances

from gensim.similarities import SoftCosineSimilarity

import csv

from scipy.spatial.distance import squareform
from sklearn.cluster import AgglomerativeClustering

import time
import emoji
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

import scipy.cluster.hierarchy as shc
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.stem.wordnet import WordNetLemmatizer
from spacy.lang.en import English
import string
import sys
sys.setrecursionlimit(10000)
from sklearn.preprocessing import normalize
from urllib.parse import urlparse

en_stop = set(nltk.corpus.stopwords.words('english'))


def tokenize(text):
    # print(text)
    def fix_str( inp ):
        ret = emoji.demojize(inp)
        ret = ret.replace( "_", " " )
        ret = ret.replace( "18+", " fuck sex " )
        ret = ret.replace( "$", " dollar money " )
        ret = ret.replace("#1", " best ")
        ret = ret.replace("-", " ")
        return ret
    
    def get_lemma(word):
        w =  WordNetLemmatizer().lemmatize(word.lower(),'v')
        return w

    text = fix_str(text)
    parser = English()
    lda_tokens = []
    text = text.translate( string.punctuation)
    tokens = text.split()
    # print(tokens)
    final_tokens=set()
    tokens = [token for token in tokens if token not in en_stop]
    for token in tokens:
        # print(token)
        try:            
            w = get_lemma(token)            
            final_tokens.add(w)
        except Exception as e:
            print(e)
            continue
    return ' '.join(final_tokens)


def get_upper_trianlge(arr):
    return arr[np.triu_indices( len(arr) , k=1 )]


file_path='data/final_data_desk_mob_12262.csv'

df_notifications =pd.read_csv(file_path, header=0)
df_notifications = df_notifications.fillna('empty')
df_notifications['text'] =  df_notifications[['title', 'body']].apply(lambda x: ' '.join(x),axis=1)
body_list = [ tokenize(str(x)) for x in df_notifications['text'] ] 
# Convert to list
data = body_list
# Remove Emails
data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]
# Remove new line characters
data = [re.sub('\s+', ' ', sent) for sent in data]
# Remove distracting single quotes
data = [re.sub("\'", "", sent) for sent in data]


url_paths=[]
for url in df_notifications['landing_url']:
    u = []
    if pd.isnull(url):
        url_paths.append('')
    else:        
        url_parsed = urlparse(url)
        u.append(url_parsed.path)
        if url_parsed.query:
            for q in url_parsed.query.split('&'):
                if '=' in q:
                    u.append(q.split('=')[0])
        url_paths.append(' '.join(u))

if 'url_path' not in df_notifications.columns:
    df_notifications['url_path'] = url_paths
    df_notifications.to_csv(file_path)


def compute_msg_dist_matrix(data):
    lst_notifications = data 
    # print(lst_notifications)
    model = Word2Vec(lst_notifications, min_count=1)  # train word-vectors
    termsim_index = WordEmbeddingSimilarityIndex(model.wv)
    data_2 = [d.split() for d in lst_notifications]
    #print(data)
    dictionary = Dictionary(data_2)
    bow_corpus = [dictionary.doc2bow(document) for document in data_2]
    similarity_matrix = SparseTermSimilarityMatrix(termsim_index, dictionary)  # construct similarity matrix
    docsim_index = SoftCosineSimilarity(bow_corpus, similarity_matrix)
    sof_cosine_distance_matrix = 1- np.array(docsim_index)
    return sof_cosine_distance_matrix

def compute_path_dist_matrix(data):
    vectorizer = CountVectorizer(data)
    vectorizer.fit(data)
    vect = vectorizer.transform(data).toarray()

    n= len(data)
    sim_mat = cosine_similarity(vect,dense_output=False)
    print(sim_mat)
    dist_mat = 1- np.array(sim_mat)
    dist_mat[dist_mat<0] = 0
    print(dist_mat)
    return dist_mat

lst_path = [x.replace('/','no path') if len(x)<3 else x for x in df_notifications['url_path']]
lst_path_print = [x.replace('/','no path')  for x in df_notifications['url_path'] if len(x)<3]

sim_matrix_path = compute_path_dist_matrix(lst_path)
sim_matrix_text = compute_msg_dist_matrix(data)
sim_matrix = sim_matrix_text + sim_matrix_path
sim_matrix= sim_matrix/2
upper_dists = get_upper_trianlge(sim_matrix)


plt.figure(figsize=(10, 7))
plt.title("Average Text Dendograms")
dend = shc.dendrogram(shc.linkage( upper_dists , method='average'))
# plt.show()

method= 'average'
from sklearn import metrics
points_average = [] 
labels_average= []
values_in_range = []
n_clusters_average = []
the_range = np.arange( 0.05,1,0.05 )
for x in the_range:
    values_in_range.append( x )
    cluster_topics = AgglomerativeClustering(n_clusters=None, affinity='precomputed', linkage=method, distance_threshold=x)
    model_topics = cluster_topics.fit(sim_matrix)    
    labels_average.append(  model_topics.labels_ )    
    n_clusters_average.append( model_topics.n_clusters_ )    
    score = metrics.silhouette_score(sim_matrix, model_topics.labels_ , metric='precomputed')
    print(x,score)
    points_average.append( [ x, score ] )
    df_notifications['cluster_'+str(x)]=model_topics.labels_

df_notifications.to_csv('data_silhoutte_scores.csv')

plt.plot( [x[0] for x in points_average],[x[1] for x in points_average] )
plt.title( 'average method/ title+body hirarchical model silhoutte score' )
plt.xlabel( 'cut off threshold' )
plt.ylabel('Silhoutte Score')
plt.show()
