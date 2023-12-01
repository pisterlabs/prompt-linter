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


# data needed for coherence calculation

# import entire dataset
f = open('coherence_vars20.sav', 'rb')

[corpus, id2word, docs] = pickle.load(f)
f.close()

# corpus - word frequency in docs
# id2word - dictionary
# docs - df["final_frqwds_removed"]

# input needed for LDA, NMF and LSA (all from Scikit-Learn) is one string per document (not a list of strings)

text = []

for abstract in docs:
    text.append(" ".join(abstract))
    

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
            
        if top_n == -1:   
            topic_words.append([vectorizer.get_feature_names()[i] for i in topic.argsort()[::-1]])
        else:
            topic_words.append([vectorizer.get_feature_names()[i] for i in topic.argsort()[:-top_n - 1:-1]])
        
    return topic_words


# create document-term matrix

vectorizer = CountVectorizer(max_df=0.6, min_df=20, lowercase=False, max_features=int(len(docs)/2))
doc_term_matrix = vectorizer.fit_transform(text)


# function adapted from https://datascienceplus.com/evaluation-of-topic-modeling-topic-coherence/

def lda_metrics(doc_term_matrix, n_topics, vectorizer, corpus, id2word, docs, rand_start):
    """
    Compute perplexity and c_v topic coherence for various number of topics

    Parameters:
    ----------
    doc_term_matrix
    n_topics : list of number of topics

    Returns:
    -------
    coherence_values : c_v topic coherence values corresponding to the LDA model with respective number of topics
    """
    
    perplexity_values = []
    coherence_values = []
    
    i = rand_start 
    for num_topics in n_topics:
        
        # create model
        t1 = time.time()
        lda_model = LatentDirichletAllocation(n_components=num_topics, doc_topic_prior = 1/num_topics, 
                                              topic_word_prior=0.1, n_jobs=39, random_state = i)
        lda_model.fit_transform(doc_term_matrix)
        t2 = time.time()
        print(f"  Model time: {t2-t1}")
        
        # compute perplexity
        perplexity_values.append(lda_model.bound_)
        
        # create list of topics
        topics = list_topics(lda_model, vectorizer, top_n=10)
        
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

    return perplexity_values, coherence_values


# code copied from https://datascienceplus.com/evaluation-of-topic-modeling-topic-coherence/
# minor alterations made
 
n_topics = list(range(5,131,5)) + [140, 150, 175, 200]
num_runs = 2

batch= 6

col_names = [f"iteration {i+batch}" for i in range(num_runs)]
lda_p = pd.DataFrame(index = n_topics, columns = col_names)
lda_c = pd.DataFrame(index = n_topics, columns = col_names)

for i in range(num_runs):
    
    print(f"Iteration {i}")
    
    # run models
    [p, c] = lda_metrics(doc_term_matrix=doc_term_matrix, n_topics=n_topics, vectorizer=vectorizer, 
                         corpus=corpus, id2word=id2word, docs=docs, rand_start = (i+batch)*len(n_topics)) # rand_start -- 228
    
    # save results
    lda_p[f"iteration {i+batch}"] = p
    lda_c[f"iteration {i+batch}"] = c
       
        
# save results 

lda_p.to_pickle("./results/final_10_runs/lda_p6-7.pkl")
lda_c.to_pickle("./results/final_10_runs/lda_c6-7.pkl")