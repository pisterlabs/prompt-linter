# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 11:13:35 2019

@author: areej
"""


#script that learns latent topics of data using LDA and the resulted topics can be improved by reranking topic terms in varios raking methods. 

#steps:
#1. Preprocess data
#2. Train an LDA model on the processed data
#3. Rerank topic words




from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary


# methods file
import methods

#parameters
NUMTOPICS = 10 # you can experiment with this parameter
NUMTERMS = 10  # you can experiment with this parameter
rank_method ="rank_idf" # choose one of the following ranking methods (rank_norm, rank_tfidf, rank_idf) refer to paper for details.
coh_metric ="c_npmi"

#input
data_file="data.txt"





###### Main ########

## prep data
raw_texts = methods.load_data(data_file)

processed_texts = methods.clean_data(raw_texts)

dictionary, corpus = methods.get_dict(processed_texts)

## create LDA model 
ldamodel = LdaModel(corpus, id2word= dictionary, num_topics= NUMTOPICS, passes=10) 

# examine learned topics
topics_list=[]
for topic_ind in range(NUMTOPICS):
    topic = ldamodel.get_topic_terms(topic_ind, NUMTERMS)
    topics_list.append([dictionary[pair[0]] for pair in topic])
    print("Topic", topic_ind,":", topics_list[topic_ind])
    
# average coherence of the learned topics
#since we filtered the dictionary, some words in the processed texts are not in the dictionary. We will create a new dictionary for coherence use only
dictionary_coh = Dictionary(processed_texts)
coh = CoherenceModel(topics = topics_list, texts = processed_texts, dictionary = dictionary_coh, coherence= coh_metric).get_coherence()
print("-" * 10 )
# Coherence will be small since the data we are using here is small and will not produce representative topics. 
print("(Ranked using Rank_orig) Topics Coherence Score %r %r \n"%(coh_metric, coh)) 

######### Start Terms Re-Ranking Here ############
## Rerank topic terms to improve thier interpretability
## First get topic_term matrix, shape (NumTopics X DictionaryLength)
topics = ldamodel.state.get_lambda()  
topics_norm= topics / topics.sum(axis=1)[:, None]
#print("topic/term distribution size ", np.shape(topics))


##
if rank_method == "rank_norm":
    # get topics from lda model then rerank terms and display new topics
    new_topics = methods.rank_norm(topics_norm, NUMTOPICS, NUMTERMS,ldamodel, dictionary)
     
if rank_method == "rank_tfidf":
    new_topics = methods.rank_tfidf(topics_norm, NUMTOPICS, NUMTERMS,ldamodel, dictionary)
    
if rank_method == "rank_idf":
    new_topics = methods.rank_idf(topics_norm, NUMTOPICS, NUMTERMS,ldamodel, dictionary, processed_texts)


    
# Create new topics by sorting the terms of each topic by new weight and get the top-n terms with the highest probability to represent the topic     
new_topics = methods.create_new_topics(new_topics, NUMTOPICS, NUMTERMS)
    
#display new topics
list1= methods.get_all_topics_reweighted_with_matrix(new_topics,  NUMTOPICS, NUMTERMS, dictionary)


new_topics_list=[]
for topic_ind in range(0, len(list1)):
    print("Topic", topic_ind,":", end=' ')
    topic = list1[topic_ind]
    temp_topic=[]
    #df = pd.DataFrame(topic, columns=['Term', 'Probability'])
    #print(df)
    for term_ind in range(0,len(topic)):
        (term, prob) = topic[term_ind]
        
        temp_topic.append(term)
       
    new_topics_list.append(temp_topic)
    print(new_topics_list[topic_ind])
    
coh = CoherenceModel(topics = new_topics_list, texts = processed_texts, dictionary = dictionary_coh, coherence=coh_metric).get_coherence()
print("-" * 10 )
print("(Ranked using %r) Topics Coherence Score %r %r "%(rank_method, coh_metric, coh))
