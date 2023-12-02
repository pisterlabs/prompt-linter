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
    
    
def print_topics(model, vectorizer, top_n=10):
    for idx, topic in enumerate(model.components_):  # loop through each row of H.  idx = row index.  topic = actual row
        print("\nTopic %d:" % (idx))
        #print([(vectorizer.get_feature_names()[i], topic[i])  # printing out words corresponding to indices found in next line
                        #for i in topic.argsort()[:-top_n - 1:-1]])  # finding indices of top words in topic
            
        print_list = [(vectorizer.get_feature_names()[i], topic[i])  
                        for i in topic.argsort()[:-top_n - 1:-1]]
        for item in print_list:
            print(item)

def list_topics(model, vectorizer, top_n=10):

    #input. top_n: how many words to list per topic.  If -1, then list all words.
       
    topic_words = []
    
    for idx, topic in enumerate(model.components_):  # loop through each row of H.  idx = row index.  topic = actual row
            
        if top_n == -1:   
            topic_words.append([vectorizer.get_feature_names()[i] for i in topic.argsort()[::-1]])
        else:
            topic_words.append([vectorizer.get_feature_names()[i] for i in topic.argsort()[:-top_n - 1:-1]])
        
    return topic_words


f = open('coherence_vars_full.sav', 'rb')

[corpus, id2word, docs] = pickle.load(f)
f.close()

text = []

for abstract in docs:
    text.append(" ".join(abstract))
    
tfidf_vectorizer = TfidfVectorizer(max_df=0.6, min_df=20, lowercase=False, max_features=int(len(docs)/2))
tf_idf = tfidf_vectorizer.fit_transform(text)

num_topics = 100

t1 = time.time()
nmf_model = NMF(n_components=num_topics, random_state = 0)
doc_topic = nmf_model.fit_transform(tf_idf)
t2 = time.time()
print(f"  Model time: {t2-t1}")
topic_term = nmf_model.components_

# calculate topic coherence

# create list of topics
topics = list_topics(nmf_model, tfidf_vectorizer, top_n=10)

t1 = time.time()
cm = CoherenceModel(topics=topics, corpus=corpus, dictionary=id2word, texts=docs, 
                    coherence='c_v', processes=10) #window_size=500 ) 
print(cm.get_coherence())
t2 = time.time()
print(f"  Coherence time: {t2-t1}")

topics_5 = list_topics(nmf_model, tfidf_vectorizer, top_n=5)

nmf_output = pd.DataFrame(cm.get_coherence_per_topic(with_std=True))
nmf_output.insert(0, 'topic_words', topics_5)
nmf_output.columns = ['topic_words', 'coherence_mean', 'coherence_stdev']

doc_topic_df = pd.DataFrame(data=doc_topic.copy())
nmf_output["avg_weight_in_corpus"] = doc_topic_df.mean(axis=0)
nmf_output["med_weight_in_corpus"] = doc_topic_df.median(axis=0)

# create a column for the number of documents that contain a topic
doc_topic_bool = pd.DataFrame(data=doc_topic.copy())
doc_topic_bool[doc_topic_bool > 0] = 1 

nmf_output["num_docs_containing_topic"] = doc_topic_bool.sum(axis=0)
nmf_output["percent_docs_containing_topic"] = 100*(nmf_output["num_docs_containing_topic"]/doc_topic.shape[0])

# find the dominant topic per document
max_topic = doc_topic_df.idxmax(axis=1)

nmf_output["num_times_max_topic"] = max_topic.value_counts()
nmf_output["percent_times_max_topic"] = 100*(nmf_output["num_times_max_topic"]/doc_topic.shape[0])

# save to file
pickle.dump([doc_topic, topic_term, nmf_output], open('nmf_tuning/full/nmf_100.sav','wb'))