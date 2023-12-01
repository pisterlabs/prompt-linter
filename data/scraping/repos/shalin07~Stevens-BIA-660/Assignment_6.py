#!/usr/bin/env python
# coding: utf-8

# In[2]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.cluster import KMeansClusterer, cosine_distance
from sklearn.decomposition import LatentDirichletAllocation

import gensim
from gensim import corpora
from gensim.models.coherencemodel import CoherenceModel

import pandas as pd
from sklearn import metrics
import numpy as np
import json, time
from matplotlib import pyplot as plt
from termcolor import colored

def cluster_kmean(train_file, test_file):

    train_file = json.load(open(train_file, 'r'))
    test_file = json.load(open(test_file, 'r'))

    tfidf_vect = TfidfVectorizer(stop_words="english",min_df=5) 
    test_text,label=zip(*test_file)
    first_label=[item[0] for item in label]
    dtm = tfidf_vect.fit_transform(train_file+list(test_text))

    num_clusters=3
    clusterer = KMeansClusterer(num_clusters, cosine_distance, repeats=20)

    clusters = clusterer.cluster(dtm.toarray(), assign_clusters=True)

    confusion_df=pd.DataFrame(list(zip(first_label, clusters[len(train_file):])), columns=['actual_class','cluster'])

    confusion_matrix = pd.crosstab( index=confusion_df.cluster, columns=confusion_df.actual_class)
    print(confusion_matrix)

    matrix = confusion_matrix.idxmax(axis=1)
    for idx, i in enumerate(matrix):
        print("Cluster {}: Topic {}".format(idx, i))

    cluster_dict={0:"Topic Travel & Transportation",                  1:"Topic Disaster and Accident",                  2:"Topic News and Economy"}

    predicted_target=[matrix[i] for i in clusters[len(train_file):]]

    print(metrics.classification_report(first_label, predicted_target))
    
def cluster_lda(train_file, test_file):

    train_file = json.load(open(train_file,'r'))
    test_file = json.load(open(test_file, 'r'))


    tf_vectorizer = CountVectorizer(min_df=5, stop_words='english')
    test_text,label=zip(*test_file)
    first_label=[item[0] for item in label]
    tf = tf_vectorizer.fit_transform(train_file+list(test_text))

    num_clusters=3

    lda = LatentDirichletAllocation(n_components=num_clusters, evaluate_every = 1, max_iter=25,verbose=1, n_jobs=1,
                                    random_state=0).fit(tf[0:len(train_file)])

    topic_assign=lda.transform(tf[len(train_file):])
    topic=topic_assign.argmax(axis=1)
    confusion_df=pd.DataFrame(list(zip(first_label, topic)), columns=['actual_class','topic'])

    confusion_matrix = pd.crosstab( index=confusion_df.topic, columns=confusion_df.actual_class)
    print(confusion_matrix)


    matrix = confusion_matrix.idxmax(axis=1)
    for idx, t in enumerate(matrix):    
        print("Cluster {}: Topic {}".format(idx, t))
        
    cluster_dict={0:"Topic Travel & Transportation",                  1:"Topic Disaster and Accident",                  2:"Topic News and Economy"}

    predicted_target=[matrix[i] for i in topic]

    print(metrics.classification_report(first_label, predicted_target))
    
    
if __name__ == "__main__":
    
    print(colored("Output of Kmeans model", 'blue', attrs=['bold']))
    cluster_kmean("train_text.json","test_text.json")
    print(colored("Output of LDA model", 'blue', attrs=['bold']))
    cluster_lda("train_text.json","test_text.json")


# In[ ]:




