#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import json
import numpy
import re
import os
import numpy as np
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from gensim.models import Doc2Vec



def load_data(input_file):
    '''
    input: result_13k.json
    output: rep_list,dep_list
    '''
    with open(input_file) as f:
        data = json.load(f)
    print(f'repos nums: {len(data)}')

    need_to_remove = []
    for k,v in data.items():
        if 'No dependency' in v:
            need_to_remove.append(k)
    print(f'repos which have no dependency files: {len(need_to_remove)}')

    for k in need_to_remove:
        del data[k]
    print(f'repos with dependency files: {len(data)}')

    rep_list,dep_list = [],[]
    for k,v in data.items():
        rep_list.append(k)
        dep_list.append(v)
        
    dep_dict = {}
    for deps in data.values():
        for i in deps:
            dep_dict[i] = dep_dict.get(i,0)+1

    print(f'distinct dependency file: {len(dep_dict)}')

    return rep_list,dep_list,data
    ### rep_list format :  ['https://github.com/AgriculturalModelExchangeInitiative/Crop2ML'  ... ]
    ### dep_list format: [['ipython', 'jupyter-sphinx', 'nbformat', 'nbsphinx', 'path-py', 'six', 'sphinx', 
    #                      'sphinx-hoverxref', 'sphinx-rtd-theme'], ['pypng', 'requests'],  ....] 
    ### data format: {repo1: [dep1,dep2], ...}

def d2v(dep_list):
    LabeledSentence1 = gensim.models.doc2vec.TaggedDocument
    all_content_train = []
    j=0
    for em in dep_list:
        all_content_train.append(LabeledSentence1(em,[j]))
        j+=1
    d2v_model = Doc2Vec(all_content_train, 
                    size = 100, 
                    window = 10, 
                    min_count = 1, 
                    workers=7, 
                    dm = 1,
                    alpha=0.025, 
                    min_alpha=0.001)
    d2v_model.train(all_content_train, 
                    total_examples=d2v_model.corpus_count, 
                    epochs=10, 
                    start_alpha=0.002, 
                    end_alpha=-0.016)

    return d2v_model
    ### d2v_model can be seen as a list, each item represents a doc vector 

def kmeans(k,d2v_model,rep_list):
    kmeans_model = KMeans(n_clusters=k, init='k-means++', max_iter=500) 
    X = kmeans_model.fit(d2v_model.docvecs.doctag_syn0)
    labels=kmeans_model.labels_

    topic_dict = {}
    for index,label in enumerate(labels):
        topic_id = label
        # print(topic_id, '--->', rep_list[index])
        topic_dict[label] = topic_dict.get(label,[])
        topic_dict[label].append(rep_list[index])

    for k in sorted(topic_dict.keys()):
        print(f'topic {k} : repos num: {len(topic_dict[k])}')

    return topic_dict
    ## topic_dict is a dictionary whose key is the topic and value is a list of repos
    ## format {top1:  [repo1,repo2] ....}

def gmm(k,d2v_model):
    GMM = GaussianMixture(n_components=k).fit(d2v_model.docvecs.doctag_syn0)
    probs = GMM.predict_proba(d2v_model.docvecs.doctag_syn0)
    #probs.shape,probs
    return probs

### LDA ### 
def LDA(data,rep_list):
    # based on dep file names , build dep name dictionary
    id2word  = corpora.Dictionary(list(data.values()))   # {0: 'emd-signal',1: 'numpy', 2: 'SQLAlchemy' ...}
    # based on dep name dict and dep names, build corpus
    corpus = [id2word.doc2bow(text) for text in list(data.values())] # [[(0, 1), (1, 1)],.....]
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=10, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)
    # pprint(lda_model.print_topics())

    print('Perplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.
    # Compute Coherence Score
    coherence_model_lda = CoherenceModel(model=lda_model, texts=list(data.values()), dictionary=id2word, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('Coherence Score: ', coherence_lda)

    # Show the top 5 words of each topic
    for topic in lda_model.print_topics(num_words=5):
        print(topic)

    # get the possible of each topic
    probs = lda_model.inference(corpus)[0]

    # inference
    topic_dict = {}
    for e, values in enumerate(lda_model.inference(corpus)[0]):
        topic_val = 0
        topic_id = 0
        for tid, val in enumerate(values):
            if val > topic_val:
                topic_val = val
                topic_id = tid        
        topic_dict[topic_id] = topic_dict.get(topic_id,[])
        topic_dict[topic_id].append(rep_list[e])

    return probs,topic_dict

