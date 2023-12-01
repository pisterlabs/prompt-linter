import warnings
from bs4 import BeautifulSoup
from bs4.element import Comment
import urllib3
import re
import nltk
import numpy as np
import pandas as pd
from pprint import pprint

# NLTK Stop words
from nltk.corpus import stopwords

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.models import Word2Vec

# Sentence transformer
from sentence_transformers import SentenceTransformer

# spacy for lemmatization
import spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim_models  # don't skip this
import matplotlib.pyplot as plt

# Enable logging for gensim - optional
import logging
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

# Removing Stop Words

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Class that creates Word2Vec vector representations for all training data used.  The vectors will be 256 length long and will have a window of length 7.
# Once the vectors are created, the vectors for each token in a given document are averaged together, producing an average vector of 256 length for each
# document.


class transformerClass:

	# Initialize the urllib reader for reading in the pages, stopwords for filtering out in the text, dataframe to store the text.  Also, read in
	# the text and store all text in an overall variable in preparation for construction of the word2vec model.
	def __init__(self, file):
		self.http = urllib3.PoolManager()

		with open(file) as f:
    			lines = f.read().splitlines()
		self.actors = lines

		self.stop_words = stopwords.words('english')
		self.stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

		self.df = pd.DataFrame()
		self.article_text = ''
		self.overallAverageVectors = []
		self.transformerModel = SentenceTransformer(
		    r"sentence-transformers/all-distilroberta-v1")

		for actor in self.actors:
		  r = self.http.request('GET', actor)
		  self.article_text += self.text_from_html(r.data)

	def tag_visible(self, element):
    		if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        		return False
    		if isinstance(element, Comment):
        		return False
    		return True

	# Create text reader using BeautifulSoup in order to filter out HTML tags and have only text data in preparation for modeling.
	def text_from_html(self, body):
    		soup = BeautifulSoup(body, 'html.parser')
    		texts = soup.findAll(text=True)
    		visible_texts = filter(self.tag_visible, texts)
    		return u" ".join(t.strip() for t in visible_texts)

	# Preprocess the text by lower casing it, removing special characters, tokenizing it, and removing stop words.

	def preprocessText(self):
		# Cleaing the text
		processed_article = self.article_text.lower()
		processed_article = re.sub('[^a-zA-Z]', ' ', processed_article)
		processed_article = re.sub(r'\s+', ' ', processed_article)

		# Preparing the dataset
		all_sentences = nltk.sent_tokenize(processed_article)

		self.all_words = [nltk.word_tokenize(sent) for sent in all_sentences]

		for i in range(len(self.all_words)):
    			self.all_words[i] = [w for w in self.all_words[i]
    			    if w not in stopwords.words('english')]

	# Use prepared text to construct word2vec model.
	# def calculateTransformerVec(self):
	#	self.word2vec = Word2Vec(self.all_words, size=256, window=7, min_count=2)
	#	self.word_vectors = self.word2vec.wv

	# Once model is constructed, get the average vector for each document.
	def getTransformerVec(self):
		for actor in self.actors:
			self.num_vectors = 0
			self.averageVector = [0] * 768
			r = self.http.request('GET', actor)
			article_text = self.text_from_html(r.data)

			# Cleaning the text
			processed_article = article_text.lower()
			processed_article = re.sub('[^a-zA-Z]', ' ', processed_article)
			processed_article = re.sub(r'\s+', ' ', processed_article)

			# Preparing the dataset
			all_sentences = nltk.sent_tokenize(processed_article)
			print(all_sentences)

			for i in all_sentences:
				if (len(i) <= 512):
					self.averageVector += self.transformerModel.encode(i)
					self.num_vectors += 1
				else:
					chunks = int(len(i)/512)
					for j in range(chunks):
						temp = i[j*512:(j+1)*512]
						self.averageVector += self.transformerModel.encode(temp)
						self.num_vectors += 1

					temp = i[chunks * 512:]
					self.averageVector += self.transformerModel.encode(temp)
					self.num_vectors += 1


			self.averageVector = [x / self.num_vectors for x in self.averageVector]
			self.overallAverageVectors.append(list(self.averageVector))

		return self.overallAverageVectors



