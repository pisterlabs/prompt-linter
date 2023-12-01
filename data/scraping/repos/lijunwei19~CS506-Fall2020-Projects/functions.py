import nltk
import numpy as py
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer 
lemmatizer = WordNetLemmatizer()

import re
import numpy as np
import pandas as pd
from pprint import pprint

# spacy for lemmatization
import spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from collections import Counter
from gensim.test.utils import datapath
import pickle


#nltk.download('stopwords')
#nltk.download('punkt')
#nltk.download('wordnet')

def save_to_csv(file_name):
	data = pd.io.stata.read_stata(file_name + '.dta')
	data.to_csv(file_name + '.csv')

def extract_label_and_feature_column(file_name):
	df = pd.read_csv(file_name)
	return df['WHO_WAS_CONTACTED'], df['COMMENTS_CLOB']

def numerize_label(string):
	if string != string:
		return 0
	if 'Client' in string:
		return 1
	if 'Client' not in string and 'collateral' in string:
		return 3
	if 'Client' not in string:
		return 2
		
def process_label_and_feature(df,X):
	df = df.apply(lambda x: numerize_label(x))
	na_values = get_na_values(X)
	df = df.drop(na_values)
	X = X.drop(na_values)
	return df,X

def get_na_values(X):
	return np.where(X.isna())[0].tolist()

def clear(x):
	stop_words = set(stopwords.words('english'))
	x=str(x)
	x=x.lower()
	word_tokens=nltk.word_tokenize(x)
	word_tokens=[lemmatizer.lemmatize(x) for x in word_tokens]
	filtered_sentence = [w for w in word_tokens if not w in stop_words]  
	return filtered_sentence 


def check_token(X):
	all_token=[]
	for x in X:
		all_token += x
		print(Counter(all_token).most_common(100))

def update_stop_words(addlist):
	stop_words = set(stopwords.words('english'))
	for x in addlist:
		stop_words.add(x)
	return stop_words

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts, stop_words):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts, bigram_mod):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts, trigram_mod, bigram_mod ):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(nlp, texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


def process_X(X):
	addlist=['.',',','cw','spoke','!','?']
	stop_words = update_stop_words(addlist)
	data = X.values.tolist()
	data_words = list(sent_to_words(data))
	# Build the bigram and trigram models
	bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
	trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  
	# Faster way to get a sentence clubbed as a trigram/bigram
	bigram_mod = gensim.models.phrases.Phraser(bigram)
	trigram_mod = gensim.models.phrases.Phraser(trigram)
	print("Finish bigram")
	print("-------------------------------------------")
	# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
	nlp = spacy.load('en',disable=['parser', 'ner']) 	# python3 -m spacy download en
	data_words_nostops = remove_stopwords(data_words, stop_words) 	# Remove Stop Words
	data_words_bigrams = make_bigrams(data_words_nostops, bigram_mod)	 	# Form Bigrams
	data_lemmatized = lemmatization(nlp, data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']) 	# Do lemmatization keeping only noun, adj, vb, adv
	id2word = corpora.Dictionary(data_lemmatized)
	# Create Corpus
	texts = data_lemmatized
	# Term Document Frequency
	corpus = [id2word.doc2bow(text) for text in texts]
	print("Finish building corpus")
	print("-------------------------------------------")
	lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=4, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)

	return corpus, lda_model


def save_model(corpus, model):
	model.save('lda.model')
	with open("corpus.txt", "wb") as p:   #Pickling
		pickle.dump(corpus, p)
	






