# Importing built-in libraries (no need to install these)
import re
import os
from os import listdir
from os.path import isfile, join

# Importing libraries you need to install
import pandas as pd
import numpy as np
from tqdm import tqdm
import itertools as it

import nltk
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 

import spacy
from spacy.lemmatizer import Lemmatizer
from spacy.lang.en.stop_words import STOP_WORDS

from gensim.corpora import Dictionary, MmCorpus
from gensim.models import Phrases
from gensim.models.word2vec import LineSentence
from gensim.parsing.preprocessing import remove_stopwords
from gensim.models.ldamulticore import LdaMulticore
from gensim.models import CoherenceModel

from wordcloud import WordCloud,STOPWORDS

import matplotlib.pyplot as plt 
import pyLDAvis
import pyLDAvis.gensim
import warnings
from pyLDAvis import PreparedData



def punct_space(token):
	"""
	helper function to eliminate tokens
	that are pure punctuation or whitespace
	"""
	
	return token.is_punct or token.is_space 


def line_review(filename):
	"""
	generator function to read in reviews from the file
	and un-escape the original line breaks in the text
	"""
	
	with open(filename, encoding='utf_8') as f:
		for text in f:
			#text_re_stop = remove_stopwords(text)
			yield text.replace('\\n', '\n')
			
def lemmatized_sentence_corpus(filename,nlp):
	"""
	generator function to use spaCy to parse reviews,
	lemmatize the text, and yield sentences
	"""
	
	for parsed_review in nlp.pipe(line_review(filename),batch_size=100, n_process=4):
		for sent in parsed_review.sents:
			print("**************************")
			yield u' '.join([token.lemma_ for token in sent if not punct_space(token)])
	print("##################################")


def trigram_bow_generator(filepath,dictionary):
	"""
	generator function to read reviews from a file
	and yield a bag-of-words representation
	"""
		
	for text in LineSentence(filepath):
		yield dictionary.doc2bow(text)


def explore_topic(lda,topic_number, topn=20):
	"""
	accept a user-supplied topic number and
	print out a formatted list of the top terms
	"""
		
	print (u'{:20} {}'.format(u'term', u'frequency') + u'\n')

	for term, frequency in lda.show_topic(topic_number, topn=20):
		print (u'{:20} {:.3f}'.format(term, round(frequency, 3)))
	
		
def topic_visualizer(lda,topic_number, topn=30):
	"""
	print out a wordcloud figure of the top terms 
	for the picked toptic
	"""
	
	stop_words = set(STOPWORDS) 
	topic = lda.show_topic(topic_number,topn)
	dict_topic = dict(topic)

	cloud = WordCloud(stopwords=stop_words,
				  background_color='white',
				  width=2500,
				  height=1800,
				  max_words=topn,
				  prefer_horizontal=1.0)
	
	cloud.generate_from_frequencies(dict_topic, max_font_size=300)
	
	plt.figure(figsize = (8, 8), facecolor = None) 
	plt.imshow(cloud) 
	plt.axis("off") 
	plt.tight_layout(pad = 0) 

	plt.show() 


def prepared_data_from_dict(vis_data):
    topic_coordinates = pd.DataFrame.from_dict(vis_data['mdsDat'])
    topic_info = pd.DataFrame.from_dict(vis_data['tinfo'])
    token_table = pd.DataFrame.from_dict(vis_data['token.table'])
    R = vis_data['R']
    lambda_step = vis_data['lambda.step']
    plot_opts = vis_data['plot.opts']
    client_topic_order = vis_data['topic.order']

    return PreparedData(topic_coordinates, topic_info,
                        token_table, R, lambda_step, plot_opts, client_topic_order)





