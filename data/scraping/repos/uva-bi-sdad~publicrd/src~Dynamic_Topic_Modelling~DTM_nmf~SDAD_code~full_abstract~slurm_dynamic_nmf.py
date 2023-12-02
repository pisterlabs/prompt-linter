import pandas as pd
import numpy
import pickle
import time
import joblib
import gensim
import matplotlib.pyplot as plt

from itertools import islice
from scipy.linalg import block_diag
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.corpora import Dictionary, bleicorpus
from gensim.matutils import hellinger
from gensim.models.coherencemodel import CoherenceModel

# Create a new document term matrix using the topic distribution
def create_matrix(windows_H, windows_terms):
    """
    Create the topic-term matrix from all window topics that have been added so far.
    Parameters:
    ----------
    windows_H: windiws topic distribution of top n words
    windows_terms: windows terms used for each fiscal year
    """
    # Set a list of all terms unique terms across windows (all_terms) and the combine windows terms (all_windows_terms)
    all_windows_terms = sum(windows_terms,[])
    
    # Create a block diagonal matrix of all topics: the number of rows is the same as the length of list_terms
    M = block_diag(*windows_H)
    
    # Identify duplicated terms (columns) and sum them
    # The fastest way is to transform M into data frame with
    dfM = pd.DataFrame(data = M, columns=all_windows_terms).groupby(level=0, axis=1).sum()
    
    # Transform back the dataframe to matrix and get the variable names (in the order in the matrix) as the final all terms
    M_concat = dfM.to_numpy()
    all_terms = list(dfM.columns)
    
    
    print('--- New document-terms have been created ---')
    
    return M_concat, all_terms



# Track the dynamic of a given topic (option topic)
def track_dynamic(topic,W,windows_topic_list):
    """
    Link topics in the first stage with topic in second stage using the matrix W
    Parameters:
    ----------
    topic: topic to track the dynamic
    W: weigth matrix from the second stage
    windows_topic_list: topic list from the first stage
    """
    # For each topic from the first stage (rows) find the topic in the second stage (columns) with the higher weight
    topic_second = []
    for i, topic_first in enumerate(W):
        topic_second.append(topic_first.argmax())
        
    # Split topics classification in the first by year
    it = iter(topic_second)
    topic_first_year = [[next(it) for _ in range(size)] for size in windows_topic]
    
    # For each topic, identify the correspondance for each year
    dynamic_topic_list = []
    for y in range(0, len(year)):
        topic_year = [i for i, e in enumerate(topic_first_year[y]) if e == topic]
        dynamic_topic_list.append(topic_year)

    # Compute the list of list of topics (list of year and list of main topic)
    dynamic_topic = []
    for y in range(0, len(year)):
        dynamic_list = dynamic_topic_list[y]
        fy_topic = [windows_topic_list[y][dynamic_list[i]] for i in range(0,len(dynamic_list))] 
        dynamic_topic.append(fy_topic)
        
    # Print the result in a dataframe
    topic_print = []
    names = []

    # print the dynamic topic
    for y in range(0,len(year)):
        for t in range(0,len(dynamic_topic[y])):
            topic_print.append(dynamic_topic[y][t])
            names.append('Year_'+str(year[y])+'_'+str(t))
        
    df = pd.DataFrame (topic_print).transpose()
    df.columns = names
    
    return df, dynamic_topic_list

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

# solve an nmf model for each windows topic
def track_dynamic(topic,W,windows_topic_list):
    """
    Link topics in the first stage with topic in second stage using the matrix W
    Parameters:
    ----------

    """
    # For each topic from the first stage (rows) find the topic in the second stage (columns) with the higher weight
    topic_second = []
    for i, topic_first in enumerate(W):
        topic_second.append(topic_first.argmax())
        
    # Split topics classification in the first by year
    it = iter(topic_second)
    topic_first_year = [[next(it) for _ in range(size)] for size in windows_topic]
    
    # For each topic, identify the correspondance for each year
    dynamic_topic_list = []
    for y in range(0, len(year)):
        topic_year = [i for i, e in enumerate(topic_first_year[y]) if e == topic]
        dynamic_topic_list.append(topic_year)

    # Compute the list of list of topics (list of year and list of main topic)
    dynamic_topic = []
    for y in range(0, len(year)):
        dynamic_list = dynamic_topic_list[y]
        fy_topic = [windows_topic_list[y][dynamic_list[i]] for i in range(0,len(dynamic_list))] 
        dynamic_topic.append(fy_topic)
        
    # Print the result in a dataframe
    topic_print = []
    names = []

    # print the dynamic topic
    for y in range(0,len(year)):
        for t in range(0,len(dynamic_topic[y])):
            topic_print.append(dynamic_topic[y][t])
            names.append('Year_'+str(year[y])+'_'+str(t))
        
    df = pd.DataFrame (topic_print).transpose()
    df.columns = names
    
    return df, dynamic_topic_list

# Load the dataset.
df = pd.read_pickle("/project/biocomplexity/sdad/projects_data/ncses/prd/Paper/FR_meta_and_final_tokens_23DEC21.pkl")
df.head()

# Compute the time variable
year = df['FY'].unique()
del df

path = '/project/biocomplexity/sdad/projects_data/ncses/prd/Dynamic_Topics_Modelling/nmf_fullabstract/'
n_topics = list(range(20,61,5))


# Create a new term-document matrix: Combining all the top term from the windiws nmf
windows_topic_list = []
windows_W = []
windows_H = []
windows_terms = []

# Build the windows H matrix
  
for fy in year:
    # Upload the nmf model 
    tfidf_vectorizer = joblib.load( path+'Term_docs_'+str(fy)+'.pkl' )[1]
    (nmf_time,topics_list,W_list,H_list) = joblib.load( path+'nmf_out/windows_nmf'+str(fy)+'.pkl' )
    (model, max_coherence) = joblib.load( path+'Coherence/model_'+str(fy)+'.pkl' )
    
    # Build the list of terms for all topics (top_n) in a given fiscal year
    fy_topic_list = topics_list[model]
    
    # Get the H and W matrix for the model
    W = W_list[model]
    H = H_list[model]
    
    # select the index of terms that appear in the topics and subset the matrix H to those terms
    if hasattr(tfidf_vectorizer, 'get_feature_names'):
        terms = tfidf_vectorizer.get_feature_names()
    else:
        terms = tfidf_vectorizer
        
    # select the index of terms that appear in the topics and subset the matrix H to those terms
    topic_terms = list(set(sum(fy_topic_list,[])))
    indcol = [terms.index(i) for i in topic_terms]
    subH = H[:,indcol]
        
    # For each topic (rows) set the weigth of terms that are not listed the topic to 0.
    for i,j in enumerate(subH):
        # by row find the index of top_n terms
        indtopic = [topic_terms.index(p) for p in fy_topic_list[i]]
        notop = [k for k in range(len(topic_terms)) if k not in indtopic]
        j[notop]=0

    # append the result
    windows_topic_list.append(fy_topic_list)
    windows_W.append(W)
    windows_H.append(subH)
    windows_terms.append(topic_terms)
    
    
# Build the new document-term matrix M
(M, all_terms) = create_matrix(windows_H, windows_terms)
 
# save the new tif-idf matrix
joblib.dump((M, all_terms), path+'new_Term_docs.pkl' )


# Run am nmf model from the new document term matrix
batch = 7
(nmf_time,topics_list,W_list,H_list) = nmf_models(doc_term_matrix=M, n_topics=n_topics, vectorizer=all_terms, rand_start = (batch)*len(n_topics))

# Save the result for the second nmf
joblib.dump((nmf_time,topics_list,W_list,H_list), path+'nmf_out/second_stage.pkl' )

# Compute the coherence for the dynamic
coherence = []

# upload the result that are necessary for the coherence
topics_list = joblib.load( path+'nmf_out/second_stage.pkl' )[1]
(docs,dictionary) = joblib.load( path+'dico_docs.pkl' )
    
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
joblib.dump((index, max_value, coherence), path+'Coherence/final_model.pkl' )
    

