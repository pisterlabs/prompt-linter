import pandas as pd
import numpy
import pickle
import time
import joblib
import gensim
import matplotlib.pyplot as plt

from itertools import islice
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.corpora import Dictionary, bleicorpus
from gensim.matutils import hellinger
from gensim.models.coherencemodel import CoherenceModel

# Remove warnings
import warnings
warnings.filterwarnings("ignore")


# set all functions
# function to list topic (modified function from https://nlpforhackers.io/topic-modeling/)
def list_topics(topic_term_dist, vectorizer, top_n=10):

    #input. top_n: how many words to list per topic.  If -1, then list all words.  
    topic_words = []
    
    for idx, topic in enumerate(topic_term_dist):  # loop through each row of H.  idx = row index.  topic = actual row
            
        if top_n == -1: 
            # check if the vectorized has an attribute get_features_names. if not vectorized contains terms hasattr('abc', 'lower')
            if hasattr(vectorizer, 'get_feature_names'):
                topic_words.append([vectorizer.get_feature_names()[i] for i in topic.argsort()[::-1]])
            else:
                topic_words.append([vectorizer[i] for i in topic.argsort()[::-1]])
        else:
            if hasattr(vectorizer, 'get_feature_names'):
                topic_words.append([vectorizer.get_feature_names()[i] for i in topic.argsort()[:-top_n - 1:-1]])
            else:
                topic_words.append([vectorizer[i] for i in topic.argsort()[:-top_n - 1:-1]])
        
    return topic_words


# function to solve the nmf (modified from https://datascienceplus.com/evaluation-of-topic-modeling-topic-coherence/)
def nmf_models(doc_term_matrix, n_topics, vectorizer, rand_start):
    """
    Compute NMF model, save topics list for coherence calc
    Parameters:
    ----------
    doc_term_matrix: document-terms matrix
    n_topics: list of topics number
    vectorizer: vector of terms
    rand_start: random seed
    """
    
    nmf_time = []
    topics_list = []
    W_list = []
    H_list = []
    
    i = rand_start
    for num_topics in n_topics:

        # create model
        t1 = time.time()
        nmf_model = NMF(n_components=num_topics, random_state = i)
        nmf_model.fit_transform(doc_term_matrix)
        t2 = time.time()
        nmf_time.append(t2-t1)
        #print(f"  Model time: {t2-t1}", flush=True)
        
        # create list of topics
        topics = list_topics(nmf_model.components_, vectorizer, top_n=10)
        topics_list.append(topics)
        
        # output completion message
        i = i+1
        #print('Number of topics =', num_topics, "complete.", flush=True)
        
        # save the matrix W and H
        W = nmf_model.fit_transform(doc_term_matrix)
        W_list.append(W)
        H = nmf_model.components_
        
        # truncate the H matrix: set the weight of the non top n words to zero
        #top_n = 10
        #for idx, topic in enumerate(H):
        #    thresold = numpy.nanmin(topic[topic.argsort()[:-top_n-1:-1]])
        #    topic[topic<thresold]=0  
        H_list.append(H)

    return nmf_time, topics_list, W_list, H_list


# Load the dataset.
df = pd.read_pickle("/project/biocomplexity/sdad/projects_data/ncses/prd/Paper/FR_meta_and_final_tokens_23DEC21.pkl")
df.head()

# Compute the time variable
year = df['FY'].unique()
del df

# Select fiscal year
fy = 2009

# Find topics that maximise the coherence for each windows
path = '/project/biocomplexity/sdad/projects_data/ncses/prd/Dynamic_Topics_Modelling/nmf_fullabstract/'
n_topics = list(range(20,61,5))
batch = 7

# Load the document-terms matrix and solve an nmf model
(tf_idf,tfidf_vectorizer,docs,dictionary) = joblib.load( path+'Term_docs_'+str(fy)+'.pkl' )
(nmf_time, topics_list, W_list, H_list) = nmf_models(doc_term_matrix=tf_idf, n_topics=n_topics, vectorizer=tfidf_vectorizer, rand_start = (batch)*len(n_topics))

# Save the nmf output
joblib.dump((nmf_time,topics_list,W_list,H_list), path+'nmf_out/windows_nmf'+str(fy)+'.pkl' )
del tf_idf
del tfidf_vectorizer
del docs
del nmf_time
del topics_list
del W_list
del H_list
del dictionary
print('------- solve nmf for a year -------')

# Compute the coherence
coherence = []

# upload the result that are necessary for the coherence
topics_list = joblib.load( path+'nmf_out/windows_nmf'+str(fy)+'.pkl' )[1]
docs = joblib.load( path+'Term_docs_'+str(fy)+'.pkl' )[2]
dictionary = joblib.load( path+'Term_docs_'+str(fy)+'.pkl' )[3]
    
for t in range(0,len(n_topics)):
    term_rankings = topics_list[t]
    cm = CoherenceModel(topics=term_rankings, dictionary=dictionary, texts=docs, coherence='c_v', topn=10, processes=1)
        
    # get the coherence value
    coherence.append(cm.get_coherence())
    print("one step")
    
# find the topics that maximize the coherence
max_value = numpy.nanmax(coherence)
index = coherence.index(max_value)

print('------- solve coherence for a year -------')
    
# Save the result from the first step
joblib.dump((index, max_value), path+'Coherence/model_'+str(fy)+'.pkl' )
    