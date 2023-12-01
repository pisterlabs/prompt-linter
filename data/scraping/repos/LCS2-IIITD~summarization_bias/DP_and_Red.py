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


# Input

source_topic = 'sample_data/source.txt'
reference = 'sample_data/ref_summary.txt'

topic_path_text = open(source_topic,'r').read()
reference_text = open(reference,'r').read()


topicid = 'test'
topic= topic_path_text
summary = reference_text



stop = set(stopwords.words('english'))

regex= RegexpTokenizer(r'\w+')
lemmatizer = WordNetLemmatizer()

alpha = list(np.arange(0.01, 1, 0.05))
eta = list(np.arange(0.01, 1, 0.05))

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


def find_rel(document_1,document_2):
    rel = 0
#     print(document.shape)
#     print(reference.shape)
    for i in range(document_2.shape[1]):
        rel = rel + (document_1[0,i]*math.log2(document_2[0,i]))
    return rel

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

def chunk(in_string,num_chunks):
    chunk_size = len(in_string)//num_chunks
    if len(in_string) % num_chunks: chunk_size += 1
    iterator = iter(in_string)
    for _ in range(num_chunks):
        accumulator = list()
        for _ in range(chunk_size):
            try: accumulator.append(next(iterator))
            except StopIteration: break
        yield ''.join(accumulator)
        
# files = list(chunk(topic,3))



files = list(chunk(topic,5))

documents_list = []
document_ids = []

for d in range(len(files)-1):
    doc_string = files[d]
    document_ids.append("Document "+str(d))
    doc_string = doc_string.replace("NEWLINE_CHAR","") # Seperator of MDS corpora
    doc_string = doc_string.lower()
#         print("Document "+str(d)+" :",doc_string)
    tokens = regex.tokenize(doc_string)
    doc = []
    for j in tokens:
        if j.lower() not in stop:
            lem = lemmatizer.lemmatize(j.lower())
            doc.append(lem)
    documents_list.append(doc)


ref_summary = summary[1:len(summary)]
#     print("Summary is :",ref_summary)
tokens = regex.tokenize(ref_summary)
doc = []


reference_summary_tokens = []
for j in tokens:
    if j.lower() not in stop:
        lem = lemmatizer.lemmatize(j.lower())
        reference_summary_tokens.append(lem)
        doc.append(lem)



documents_list.append(doc)
common_dictionary = Dictionary(documents_list)
common_corpus = [common_dictionary.doc2bow(text) for text in documents_list]
n,a,e = parameter_tuning(common_corpus,common_dictionary,documents_list)
final_lda = gensim.models.LdaMulticore(common_corpus,id2word = common_dictionary,alpha = a,eta = e, workers=5,num_topics=n,chunksize=100,passes=10,random_state=100)

reference_summary_tokens = [reference_summary_tokens]

other_corpus = [common_dictionary.doc2bow(text) for text in reference_summary_tokens]
unseen_doc = other_corpus[0]
vector = final_lda[unseen_doc] 

print("Topic is :",topicid)
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


lda_array_reference = np.full((1,n),0.001)
vector_ref = final_lda[unseen_doc] 
for j in vector_ref:
    col = j[0]
    lda_array_reference[0,col] = j[1]

print("topic array :")
for i in range(lda_array.shape[0]-1):
    print(document_ids[i],":",lda_array[i:i+1,:])
#     print(np.sum(lda_array[10:11,:]))

relevance = []
for i in range(0,lda_array.shape[0]):
    document = lda_array[i:i+1,:]
    reference = lda_array[lda_array.shape[0]-1:lda_array.shape[0],:]
    cur_rel = find_rel(lda_array_reference,document)
    relevance.append(cur_rel)

redundancy = 0
ref_vector = lda_array_reference
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
#             doc_2 = lda_array[j:j+1,:]
            doc_2 = lda_array_reference
            intra_topic_r[i,j] = find_rel(doc_1,doc_2)



redundancy_vector = []
for i in range(0,lda_array.shape[0]-1):
    red = 0 
#         d_vector = lda_array[i:i+1,:]
    d_vector = lda_array_reference
#         print('###############',d_vector)
    for j in range(d_vector.shape[1]):
        red = red + (d_vector[0,j]*math.log2(d_vector[0,j]))
#             print('redundancy    ',red)
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
#     ax = sns.heatmap(normalized_intra_topic_r,vmin=-1, vmax=0 ,cmap = "YlGnBu",annot=True,linewidth=0.5)
#     plt.show()
print()

mx = maximum(intra_topic_d)
mn = minimum(intra_topic_d)
normalized_intra_topic_d = normalize_d(intra_topic_d,mn,mx)
print("Per document divergence is :")
perdoc_div = expectation(normalized_intra_topic_d)
print()
print("Intra-topic divergence is :")
#     ax = sns.heatmap(normalized_intra_topic_d,vmin=0, vmax=1 ,cmap = "YlGnBu",annot=True,linewidth=0.5)
#     plt.show()
print()

print("Redundancy vector is :")
print(redundancy_vector)
print()
redundancy_dataset.append(sum(redundancy_vector)/len(redundancy_vector))
relevance_dataset.append(sum(perdoc_rel)/len(perdoc_rel))


