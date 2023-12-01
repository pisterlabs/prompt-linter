#!/usr/bin/env python3
import pandas as pd
import numpy as np
import os
import math
import string
import re
import pprint
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import csv
import pyLDAvis
import pyLDAvis.gensim 
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark import SparkFiles
from pyspark.sql import SQLContext
from pyspark.sql import Row
from pyspark.sql import SparkSession
from nltk.corpus.reader.util import StreamBackedCorpusView
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.data import load
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('tagsets')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
import gensim
from gensim import corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
os.environ["SPARK_HOME"] = "spark-3.2.1-bin-hadoop3.2"
import spacy
spacy.load('en_core_web_sm')
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
stop_words = stopwords.words('english')

if __name__ == "__main__":
#Setup 
        #Open Poe Story
	story = open("poe-stories/THE_BLACK_CAT", "r")

        #Convert Poe Story to Lower
	story = story.read()
	story = story.lower()

	#Tokenize Story
	sent_text = nltk.sent_tokenize(story)     
	all_tagged = [nltk.pos_tag(nltk.word_tokenize(sent)) for sent in sent_text]
	print(all_tagged)
	
	#Create structure to count adjs/nouns/verbs and store counts
	adjs = [None] * 10
	nouns = [None] * 10
	verbs = [None] * 10
	adjCount = 0
	nounCount = 0
	verbCount = 0
	
	#Print first 10 adjs/nouns/verbs
	for i in range(len(all_tagged)):
		for j in range(len(all_tagged[i])):
			tagType = all_tagged[i][j][1]
			if ((tagType == "JJ" or tagType == "JJR" or tagType == "JJS") and adjCount < 10):
	            		adjs[adjCount] = all_tagged[i][j][0]
	            		adjCount+=1
			if ((tagType == "NN" or tagType == "NNS" or tagType == "NNP" or tagType == "NNPS") and nounCount < 10):
	            		nouns[nounCount] = all_tagged[i][j][0]
	            		nounCount+=1
			if ((tagType == "VB" or tagType == "VBD" or tagType == "VBG" or tagType == "VBN" or tagType == "VBP" or tagType == "VBZ") and verbCount < 10):
	            		verbs[verbCount] = all_tagged[i][j][0]
	            		verbCount+=1

	print("adjectives (10/<total adjective count>):" + str(adjs))
	print("nouns (10/<total noun count>):" + str(nouns))
	print("verbs (10/<total verb count>):" +  str(verbs))


"""
	Penn Part of Speech Tags for Reference
	7.	JJ	Adjective
	8.	JJR	Adjective, comparative
	9.	JJS	Adjective, superlative

	12.	NN	Noun, singular or mass
	13.	NNS	Noun, plural
	14.	NNP	Proper noun, singular
	15.	NNPS	Proper noun, plural
	
	27.	VB	Verb, base form
	28.	VBD	Verb, past tense
	29.	VBG	Verb, gerund or present participle
	30.	VBN	Verb, past participle
	31.	VBP	Verb, non-3rd person singular present
	32.	VBZ	Verb, 3rd person singular present


"""

#---------------------------------------------------#

#Print word in file
reviewsDict = []
            
def collect (sentences):
	speech = []
	for sent in sentences:
		for fragment in sent:
			speech.extend(fragment)
			return speech

def clean_sents(data):
	# Remove new line characters
	data = re.sub('\s+', ' ', str(data))
	data = re.sub('[(.*!@#$%^&*\'";:/?,~`+=|)]', '', str(data))
	data = data.lower()
	return data
                        
def sent_to_words(sentence):
	words = sentence.split(" ")
	for word in words:
		yield(gensim.utils.simple_preprocess(str(word).encode('utf-8'), deacc=True))  # deacc=True removes punctuations

def remove_stopwords(texts):
	return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
            
def make_bigrams(texts):
	return [bigram_mod[doc] for doc in texts]
            
def make_trigrams(texts):
	return [trigram_mod[bigram_mod[doc]] for doc in texts]
            
def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        #print(sent)
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out
                                               

colnames = ['recNo', 'ClothingID', 'Age', 'Title', 'ReviewText', 'Rating', 'ReccomendedIND', 'PositiveFeedbackCount', 'DivisionName', 'DepartmentName', 'ClassName']
reviewsDF = pd.read_csv('reviews.csv', names=colnames)

data_words = []
for review in reviewsDF.ReviewText:
	review = clean_sents(review)
	data = sent_to_words(review)
	data = [dw for dw in data if len(dw)>0]
	for datum in data:
		data_words.append(datum)
#print(data_words)
        
# Build the bigram and trigram models
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  
                        
# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)
#print(trigram_mod[bigram_mod[data_words[6]]])

# Remove Stop Words
data_words_nostops = remove_stopwords(data_words)
#print(data_words_nostops)
                        
# Form Bigrams
data_words_bigrams = make_bigrams(data_words_nostops)
#print(data_words_bigrams)
                        
# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
# python3 -m spacy download en
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

#print(len(data_words_bigrams))
# Do lemmatization keeping only noun, adj, vb, adv
data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
            
# Create Dictionary
id2word = corpora.Dictionary(data_lemmatized)
                        
# Create Corpus
texts = data_lemmatized
                        
# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]
                        
#print ([[(id2word[id], freq) for id, freq in cp] for cp in corpus])
speeches_corpus = dict(id2word)
#print(speeches_corpus)

num_topics = 10
#print(corpus)
#print(len(corpus))
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=num_topics, random_state=100, update_every=1, chunksize=100, passes=10, alpha='auto', per_word_topics=True)
print(lda_model.print_topics())



            
            
               
            
            
                        
