import csv
import nltk;
import re
import numpy as np
import pandas as pd
import gensim
import pdb
import gensim.corpora as corpora
import warnings
from gensim.models import HdpModel # from gensim.models import CoherenceModel, LdaModel, LsiModel, HdpModel
warnings.filterwarnings("ignore",category=DeprecationWarning)
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import string

stop = set(stopwords.words('english'))
exclude = set(string.punctuation) 
lemma = WordNetLemmatizer()

def clean(doc):
	stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
	punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
	normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
	return normalized

def topic_modeling(paratext,typel='hdp',numtopics=20,npasses=100):
	doc_clean = [clean(doc).split() for doc in paratext]
	dictionary = corpora.Dictionary(doc_clean)
	doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
	# 
	if(typel == 'hdp'):
		lda_model = HdpModel(corpus=doc_term_matrix, id2word=dictionary)
	elif(typel == 'lda'):
		lda_model = gensim.models.ldamodel.LdaModel(doc_term_matrix, num_topics=numtopics, id2word = dictionary, passes=npasses,minimum_probability=0.0)
	else:
		print('invalid option in topic_modeling()')
		exit(1)
	return lda_model,dictionary,doc_term_matrix;

def get_lda_embeddings(lda_model,dictionary,data):
	docdata = dictionary.doc2bow(clean(data).split())
	# doc_topics = lda_model.get_document_topics(docdata,minimum_probability=0.0)
	doc_topics = lda_model[docdata]
	topic_vec = np.array([doc_topics[i][1] for i in range(len(doc_topics))])
	return topic_vec

def get_hdp_labels(lda_model,dictionary,data):
	docdata = dictionary.doc2bow(clean(data).split())
	# doc_topics = lda_model.get_document_topics(docdata,minimum_probability=0.0)
	doc_topics = lda_model[docdata]
	prob_vec = np.array([doc_topics[i][1] for i in range(len(doc_topics))])
	label_vec = np.array([doc_topics[i][0] for i in range(len(doc_topics))])
	label = label_vec[np.argmax(prob_vec)]
	return label