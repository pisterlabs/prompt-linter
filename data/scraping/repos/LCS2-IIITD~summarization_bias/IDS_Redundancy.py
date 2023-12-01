
# coding: utf-8

# In[1]:


import gensim
import os
from gensim.corpora.dictionary import Dictionary
import nltk
import glob
from nltk import word_tokenize,WordNetLemmatizer,PorterStemmer,RegexpTokenizer
from collections import defaultdict
from nltk.util import ngrams
from nltk.corpus import stopwords
from scipy import stats
import numpy as np
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt
import math
import seaborn as sns
import sys
import random


# In[ ]:


nltk.download('wordnet')


# In[2]:


stop = set(stopwords.words('english'))


# In[4]:


print(stop)


# In[3]:


topic_path = ''
ref_path = ''


# In[4]:


regex= RegexpTokenizer(r'\w+')
lemmatizer = WordNetLemmatizer()


# In[5]:


alpha = list(np.arange(0.01, 1, 0.05))
eta = list(np.arange(0.01, 1, 0.05))


# In[7]:


print(len(alpha))


# In[6]:


def line_plot(doc_importance,labels):
#     print(doc_importance.shape)
    index = np.arange(doc_importance.shape[0])
    label = labels
    importance = list(doc_importance)
    plt.plot(index,importance, linestyle='solid')
    plt.xlabel('Sentence number', fontsize=5)
    plt.ylabel('Imprtance Value', fontsize=5)
    plt.xticks(index, label, fontsize=5, rotation=30)
    plt.title('Importance throughout document')
    plt.show()


# In[7]:


def parameter_tuning(common_corpus,common_dictionary,common_texts):
    coherence_score = []
    labels = []
    
#     print("starting number of topics.....")
    
    for i in range(2,20):
        lda = gensim.models.LdaMulticore(common_corpus,id2word = common_dictionary,workers=5,num_topics=i,chunksize=100,passes=10,random_state=100)
        coherence_model_lda = CoherenceModel(model=lda, texts=common_texts, dictionary=common_dictionary, coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
#         print("Topics :",i)
        labels.append(i)
        coherence_score.append(coherence_lda)
        
#     line_plot(np.asarray(coherence_score),labels)
    sorted_labels = [labels for _,labels in sorted(zip(coherence_score,labels))]
    n_o_t =  sorted_labels[len(sorted_labels)-1]
    
#     print("selected number of topics is:",n_o_t)
    
#     print("starting Alphas.....")
    
    coherence_score = []
    alphas = []
    for i in alpha:
        lda = gensim.models.LdaMulticore(common_corpus,id2word = common_dictionary,alpha = i , workers=5,num_topics=n_o_t,chunksize=100,passes=10,random_state=100)
        coherence_model_lda = CoherenceModel(model=lda, texts=common_texts, dictionary=common_dictionary, coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
#         print("Alpha :",i)
        alphas.append(i)
        coherence_score.append(coherence_lda)
    
#     line_plot(np.asarray(coherence_score),alphas)
    sorted_alphas = [alphas for _,alphas in sorted(zip(coherence_score,alphas))]
    alpha_val =  sorted_alphas[len(sorted_alphas)-1]
    
#     print("selected value of alpha is:",alpha_val)
    
#     print("starting Etas.....")
    
    coherence_score = []
    etas = []
    for i in eta:
        lda = gensim.models.LdaMulticore(common_corpus,id2word = common_dictionary,alpha = alpha_val,eta = i, workers=5,num_topics=n_o_t,chunksize=100,passes=10,random_state=100)
        coherence_model_lda = CoherenceModel(model=lda, texts=common_texts, dictionary=common_dictionary, coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
#         print("Etas :",i)
        etas.append(i)
        coherence_score.append(coherence_lda)
    
#     line_plot(np.asarray(coherence_score),etas)
    sorted_etas = [etas for _,etas in sorted(zip(coherence_score,etas))]
    eta_val =  sorted_etas[len(sorted_etas)-1]
#     print("selected value of eta is:",eta_val)
    return n_o_t,alpha_val,eta_val


# In[8]:


def find_rel(document_1,document_2):
    rel = 0
#     print(document.shape)
#     print(reference.shape)
    for i in range(document_1.shape[1]):
        rel = rel + (document_1[0,i]*math.log2(document_2[0,i]))
    return rel


# In[9]:


def minimum(matrix):
    rows,cols = matrix.shape
    mn = sys.maxsize
    for i in range(rows):
        for j in range(cols):
            if i!=j and matrix[i,j]<mn:
                mn = matrix[i,j]
    return mn

def maximum(matrix):
    rows,cols = matrix.shape
    mx = -sys.maxsize-1
    for i in range(rows):
        for j in range(cols):
            if i!=j and matrix[i,j]>mx:
                mx = matrix[i,j]
    return mx


# In[10]:


def normalize_r(matrix,mn,mx):
    rows,cols = matrix.shape
    for i in range(rows):
        for j in range(cols):
            if i!=j:
                matrix[i,j] = ((matrix[i,j]-mn)/(mx-mn))-1
    return matrix

def normalize_d(matrix,mn,mx):
    rows,cols = matrix.shape
    for i in range(rows):
        for j in range(cols):
            if i!=j:
                matrix[i,j] = ((matrix[i,j]-mn)/(mx-mn))
    return matrix


# In[11]:


def expectation(matrix):
    rows,cols = matrix.shape
    docs = []
    
    for j in range(cols):
        val = 0
        for i in range(rows):
            if i!=j :
                val = val + matrix[i,j]
        docs.append(val)
        
    print(docs)
    return docs


# In[19]:


def divergence(file,docs,ref_string):

    documents_list = []
    document_ids = []
    
    for d in range(len(docs)-1):
        document_ids.append(d)
        string = docs[d]
#         print("Doc :",d)
#         print(string)
        tokens = regex.tokenize(string)
        doc = []
        for j in tokens:
            if j.lower() not in stop:
                lem = lemmatizer.lemmatize(j.lower())
                doc.append(lem)
        documents_list.append(doc)
        
    string = ref_string
    tokens = regex.tokenize(string)
    doc = []
    
    for j in tokens:
        if j.lower() not in stop:
            lem = lemmatizer.lemmatize(j.lower())
            doc.append(lem)
            
    documents_list.append(doc)
    common_dictionary = Dictionary(documents_list)
    common_corpus = [common_dictionary.doc2bow(text) for text in documents_list]
    n,a,e = parameter_tuning(common_corpus,common_dictionary,documents_list)
    final_lda = gensim.models.LdaMulticore(common_corpus,id2word = common_dictionary,alpha = a,eta = e, workers=5,num_topics=4,chunksize=100,passes=10,random_state=100)
    
#     ref_vector = final_lda[common_corpus[len(common_corpus)-1]]
    
#     for i in range(len(common_corpus)):
#         print(final_lda[common_corpus[i]])
    
    print("Topic is :",file)
    print()
    
    print("topic distribution by words :")
    topic_words_dist = final_lda.show_topics(num_words=10, log=False, formatted=True)
    for i in range(len(topic_words_dist)):
        print(topic_words_dist[i])
    
    print()
    
    lda_array = np.full((len(common_corpus),n),0.001)
    for i in range(lda_array.shape[0]):
        vector = final_lda[common_corpus[i]]
        for j in vector:
            col = j[0]
            lda_array[i,col] = j[1]
            
    print("topic array :")
    for i in range(lda_array.shape[0]):
        if i!=lda_array.shape[0]-1:
            print(document_ids[i],":",lda_array[i:i+1,:])
        else:
            print("Reference summary :",lda_array[i:i+1,:])
    print()
#     print(np.sum(lda_array[10:11,:]))
    
    relevance = []
    for i in range(0,lda_array.shape[0]-1):
        document = lda_array[i:i+1,:]
        reference = lda_array[lda_array.shape[0]-1:lda_array.shape[0],:]
        cur_rel = find_rel(reference,document)
        relevance.append(cur_rel)
        
    redundancy = 0
    ref_vector = lda_array[lda_array.shape[0]-1:lda_array.shape[0],:]
    for i in range(ref_vector.shape[1]):
        redundancy = redundancy + (ref_vector[0,i]*math.log2(ref_vector[0,i]))
        
    intra_topic_r = np.zeros((lda_array.shape[0]-1,lda_array.shape[0]-1))
    r,c = intra_topic_r.shape
    for i in range(r):
        for j in range(c):
            if i==j:
                intra_topic_r[i,j] = np.inf
            else:
                doc_1 = lda_array[i:i+1,:]
                doc_2 = lda_array[j:j+1,:]
                intra_topic_r[i,j] = find_rel(doc_1,doc_2)
             
    
    redundancy_vector = []
    for i in range(0,lda_array.shape[0]-1):
        red = 0 
        d_vector = lda_array[i:i+1,:]
        for j in range(d_vector.shape[1]):
            red = red + (d_vector[0,j]*math.log2(d_vector[0,j]))
        redundancy_vector.append(red)
        
    intra_topic_d = np.zeros((lda_array.shape[0]-1,lda_array.shape[0]-1))
    r,c = intra_topic_d.shape
    for i in range(r):
        for j in range(c):
            if i==j:
                intra_topic_d[i,j] = np.inf
            else:
                intra_topic_d[i,j] = -(intra_topic_r[i,j] - redundancy_vector[i])
        
    mx = maximum(intra_topic_r)
    mn = minimum(intra_topic_r)
    normalized_intra_topic_r = normalize_r(intra_topic_r,mn,mx)
    print("Per document relevance is :")
    perdoc_rel = expectation(normalized_intra_topic_r)
    print()
    print("Intra-topic relevance is :")
    sns.set(font_scale=1.5)
    ax = sns.heatmap(normalized_intra_topic_r,vmin=-1, vmax=0 ,cmap = "YlGnBu",annot=False,linewidth=2.5)
    plt.savefig(file[0:len(file)-4]+".svg")
    plt.show()
    print()

    mx = maximum(intra_topic_d)
    mn = minimum(intra_topic_d)
    normalized_intra_topic_d = normalize_d(intra_topic_d,mn,mx)
    print("Per document divergence is :")
    perdoc_div = expectation(normalized_intra_topic_d)
    print()
    print("Intra-topic divergence is :")
    ax = sns.heatmap(normalized_intra_topic_d,vmin=0, vmax=1 ,cmap = "YlGnBu",annot=True,linewidth=0.5)
    plt.show()
    print()
    
    print("Redundancy vector is :")
    print(redundancy_vector)
    print()
    redundancy_dataset.append(sum(redundancy_vector)/len(redundancy_vector))
    relevance_dataset.append(sum(perdoc_rel)/len(perdoc_rel))
#     print("Redundancy is :",redundancy)
#     print()
#     print("Relevance list is :",relevance)
#     print()
#     print("Average Relevance is :",sum(relevance)/len(relevance))
#     print()


# In[26]:


f = '693.txt'
doc_file = open(topic_path + '/'+ f, 'r', encoding='utf-8')
ref_file = open(ref_path + '/'+ f, 'r', encoding='utf-8')
doc_string = doc_file.read()
docs = doc_string.split("\n\n")
ref_string = ref_file.read()
for d in docs:
    print("Doc :")
    print(d)
print(ref_string)
# divergence(f,docs,ref_string)


# In[ ]:


redundancy_dataset = []
relevance_dataset = []
doc_files = os.listdir(topic_path)

indices = random.sample(range(0,len(doc_files)),100)
sampled_topics = []

for i in indices:
    sampled_topics.append(doc_files[i])

for f in sampled_topics:
    doc_file = open(topic_path + '/'+ f, 'r', encoding='utf-8')
    ref_file = open(ref_path + '/'+ f, 'r', encoding='utf-8')
    doc_string = doc_file.read()
    docs = doc_string.split("\n\n")
    ref_string = ref_file.read()
    print(f)
    divergence(f,docs,ref_string)


# In[22]:


print("Dataset redundancy :",sum(redundancy_dataset)/len(redundancy_dataset))
print("Dataset relevance :",sum(relevance_dataset)/len(relevance_dataset))


# In[23]:


len(redundancy_dataset)


# In[29]:


for i in sampled_topics:
    if int(i[:len(i)-4]) > 80561 :
        print(i)


# In[20]:


f = '86354.txt'
doc_file = open(topic_path + '/'+ f, 'r', encoding='utf-8')
ref_file = open(ref_path + '/'+ f, 'r', encoding='utf-8')
doc_string = doc_file.read()
docs = doc_string.split("\n\n")
ref_string = ref_file.read()
for d in docs:
    print("Doc :")
    print(d)
print(ref_string)
divergence(f,docs,ref_string)

