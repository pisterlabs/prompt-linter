import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import gensim
import time

from sklearn.decomposition import NMF, TruncatedSVD, LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from gensim.models.coherencemodel import CoherenceModel

# DATA INGESTION =========================================================

# import NSF data
#f = open('../../data/prd/RND Topic Modelling/nsf_stanford_lemma.sav', 'rb')

# import entire dataset
f = open('../../data/prd/RND Topic Modelling/lda_data_stanford_lemma.sav', 'rb')

[corpus, id2word, docs] = pickle.load(f)
f.close()

# corpus - word frequency in docs
# id2word - dictionary
# docs - lemmatized abstracts

# input needed for LDA, NMF and LSA (all from Scikit-Learn) is one string per document (not a list of strings)

text = []
for doc in docs:
    text.append(" ".join(doc))
    
# FUNCTIONS NEEDED FOR ALL MODELS  ============================================

# function slightly modified from https://nlpforhackers.io/topic-modeling/

def print_topics(model, vectorizer, top_n=10):
    for idx, topic in enumerate(model.components_):  # loop through each row of H.  idx = row index.  topic = actual row
        print("\nTopic %d:" % (idx))
        #print([(vectorizer.get_feature_names()[i], topic[i])  # printing out words corresponding to indices found in next line
                        #for i in topic.argsort()[:-top_n - 1:-1]])  # finding indices of top words in topic
            
        print_list = [(vectorizer.get_feature_names()[i], topic[i])  
                        for i in topic.argsort()[:-top_n - 1:-1]]
        for item in print_list:
            print(item)
            

# Function to format topics as a "list of list of strings".
# Needed for topic coherence function in Gensim

# function modified from https://nlpforhackers.io/topic-modeling/

def list_topics(model, vectorizer, top_n=10):

    #input. top_n: how many words to list per topic.  If -1, then list all words.
       
    topic_words = []
    
    for idx, topic in enumerate(model.components_):  # loop through each row of H.  idx = row index.  topic = actual row
        #print("\nTopic %d:" % (idx))
        #print([(vectorizer.get_feature_names()[i], topic[i])  # printing out words corresponding to indices found in next line
                        #for i in topic.argsort()[:-top_n - 1:-1]])  # finding indices of top words in topic
            
        if top_n == -1:   
            topic_words.append([vectorizer.get_feature_names()[i] for i in topic.argsort()[::-1]])
        else:
            topic_words.append([vectorizer.get_feature_names()[i] for i in topic.argsort()[:-top_n - 1:-1]])
        
    return topic_words
    
    
# CREATE TF_IDF  ============================================================================

# used for NMF and LSA

tfidf_vectorizer = TfidfVectorizer(max_df=0.4, min_df=3, lowercase=False, max_features=int(len(docs)/2))
tf_idf = tfidf_vectorizer.fit_transform(text)


# LSA RUNS ===================================================================================

# function adapted from https://datascienceplus.com/evaluation-of-topic-modeling-topic-coherence/

def lsa_metrics(doc_term_matrix, n_topics, vectorizer, corpus, id2word, docs, rand_start):
    """
    Compute c_v topic coherence for various number of topics

    Parameters:
    ----------
    tf_idf
    n_topics : list of number of topics

    Returns:
    -------
    coherence_values : c_v topic coherence values corresponding to the LSA model with respective number of topics
    """
    
    coherence_values = []
    
    i = rand_start
    for num_topics in n_topics:

        # create model
        t1 = time.time()
        lsa_model = TruncatedSVD(n_components=num_topics, random_state = i)
        lsa_model.fit_transform(doc_term_matrix)
        t2 = time.time()
        print(f"  Model time: {t2-t1}")
        
        # create list of topics
        topics = list_topics(lsa_model, vectorizer, top_n=10)
        
        # calculate coherence
        t1 = time.time()
        cm = CoherenceModel(topics=topics, corpus=corpus, dictionary=id2word, texts=docs, 
                            coherence='c_v', processes=10) #window_size=500 ) 
        coherence_values.append(cm.get_coherence())
        t2 = time.time()
        print(f"  Coherence time: {t2-t1}")
        
        # output completion message
        i = i+1
        print('Number of topics =', num_topics, "complete.")

    return coherence_values
    
    
    
# code copied from https://datascienceplus.com/evaluation-of-topic-modeling-topic-coherence/
# minor alterations made

n_topics = range(4,101,4) 
num_runs = 5

batch = 5

col_names = [f"iteration {i+batch}" for i in range(num_runs)]
lsa_c = pd.DataFrame(index = n_topics, columns = col_names)

for i in range(num_runs):
    
    print(f"Iteration {i}")
    
    # run models
    c = lsa_metrics(doc_term_matrix=tf_idf, n_topics=n_topics, vectorizer=tfidf_vectorizer, 
                         corpus=corpus, id2word=id2word, docs=docs, rand_start = (i+batch)*len(n_topics))
    
    # save results
    lsa_c[f"iteration {i+batch}"] = c       
       
    
# SAVE RESULTS ==========================================

#lsa_c.to_pickle("./nsf_lsa_c.pkl")

lsa_c.to_pickle("./lsa_c5-9.pkl")