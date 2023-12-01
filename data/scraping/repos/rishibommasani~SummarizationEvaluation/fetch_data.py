import random
import torch
import math
# from newsroom import jsonl
import jsonlines as jsonl
from fragments import Fragments
# from newsroom.analyze.rouge import ROUGE_N
# from newsroom.analyze.rouge import ROUGE_L
import pickle
import spacy
import nltk
from nltk import word_tokenize, sent_tokenize
from tqdm import tqdm
spacy_tokenizer = spacy.load("en_core_web_sm")
nltk.download('stopwords')
from nltk.corpus import stopwords
from gensim.test.utils import common_corpus, common_dictionary
from gensim.models.wrappers import LdaMallet
path_to_mallet_binary = "Mallet/bin/mallet"
from transformers import BertTokenizer, BertForNextSentencePrediction
import re
import numpy as np
import pandas as pd
from pprint import pprint
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt
from dataset import *
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
nlp = spacy.load('en', disable=['parser', 'ner'])


def fetch_cmu(output):
	for dataset_name in output:
		print("Fetching {} Dataset".format(dataset_name))
		data = Dataset(dataset=dataset_name, emb='bert', dataset_type='train', path='./', ranges=False) 
		output[dataset_name] = data.load_dataset()
		print("Fetched {} Dataset".format(dataset_name))
	return output



def fetch_nyt(pickled=True, N=100000):
	print("Fetching NYT Dataset")
	raw_data_pickle = 'nyt_raw.pickle'
	pickle_f = '{}examples_nyt.pickle'.format(N)
	if pickled:
		data = pickle.load(open(pickle_f + str(N), 'rb'))
	else:
		documents, summaries = pickle.load(open(raw_data_pickle, 'rb'))
		assert len(documents) == len(summaries)
		raw_data = list(zip(summaries, documents))
		print("Nyt", len(raw_data))
		raw_data = raw_data[ : N]
		data = []
		for s, d in tqdm(raw_data):
			s = s[0]
			d = " ".join(d)
			f = Fragments(s, d)
			data.append({'summary' : s, 'text' : d, 'coverage' : f.coverage(), 'density' : f.density(), 'compression' : f.compression()})
		pickle.dump(data, open(pickle_f, 'wb'))
	print("Fetched NYT with {} examples".format(len(data)))
	return data


def fetch_tldr(pickled=True, N=100000):
	print("Fetching Reddit TL;DR Dataset")
	pickle_f = '{}examples_reddit.pickle'.format(N)
	if pickled:
		data = pickle.load(open(pickle_f + str(N), 'rb'))
	else:
		with jsonl.open("tldr-training-data.jsonl") as data_file:
			raw_data = list(data_file)
		print("Tldr", len(raw_data))
		raw_data = raw_data[ : N]
		
		data = []
		for ex in tqdm(raw_data):
			s, d = ex['summary'], ex['content']
			f = Fragments(s, d)
			data.append({'summary' : s, 'text' : d, 'coverage' : f.coverage(), 'density' : f.density(), 'compression' : f.compression()})
		pickle.dump(data, open(pickle_f, 'wb'))
	print("Fetched Reddit TL;DR Dataset with {} examples".format(len(data)))
	return data


def fetch_gigaword(pickled=True, N=100000):
	print("Fetching Gigaword Dataset")
	pickle_f = '{}examples_gigaword.pickle'.format(N)
	if pickled:
		data = pickle.load(open(pickle_f + str(N), 'rb'))
	else:
		with open("gigaword.article.txt") as document_file:
			documents = [d for d in document_file]
		with open("gigaword.summary.txt") as summary_file:
			summaries = [s for s in summary_file]
		raw_data = []
		for s, d in zip(summaries, documents):
			if len(s.split()) != 0 and len(d.split()) != 0:
				raw_data.append((s, d))
		print("Giga", len(raw_data))
		raw_data = raw_data[ : N]
		data = []
		for s, d in tqdm(raw_data):
			f = Fragments(s, d)
			data.append({'summary' : s, 'text' : d, 'coverage' : f.coverage(), 'density' : f.density(), 'compression' : f.compression()})
		pickle.dump(data, open(pickle_f, 'wb'))
	print("Fetched Gigaword Dataset with {} examples".format(len(data)))
	return data	


def fetch_newsroom(pickled=True, N=100000):
	print("Fetching Newsroom Dataset")
	pickle_f = '{}examples_newsroom.pickle'.format(N)
	if pickled:
		data = pickle.load(open(pickle_f + str(N), 'rb'))
	else:
		with jsonl.open("newsroom.jsonl") as data_file:
			raw_data = list(data_file)
		print("Nws", len(raw_data))
		raw_data = raw_data[ : N]
		data = [{'summary' :  e['summary'], 'text': e['text'], 'coverage' : e['coverage'], 'density' : e['density'], 'compression' : e['compression']} for e in raw_data]
		pickle.dump(data, open(pickle_f, 'wb'))
	print("Fetched Newsroom Dataset with {} examples".format(len(data)))
	return data


def fetch_cnndm(pickled=True, N=100000):
	print("Fetching CNN/DM Dataset")
	pickle_f = '{}examples_cnndm.pickle'.format(N)
	if pickled:
		data = pickle.load(open(pickle_f + str(N), 'rb'))
	else:
		with open("cnndm.docs") as document_file:
			documents = [d for d in document_file]
		with open("cnndm.summaries") as summary_file:
			summaries = [s for s in summary_file]
		print("cnndm", len(documents))
		raw_data = []
		for s, d in zip(summaries, documents):
			if len(s.split()) != 0 and len(d.split()) != 0:
				raw_data.append((s, d))
		raw_data = raw_data[ : N]
		data = []
		for s, d in tqdm(raw_data):
			s = " ".join([w for w in s.split() if w not in {'<t>', '</t>'}])
			f = Fragments(s, d, tokenize=False)
			data.append({'summary' : s, 'text' : d, 'coverage' : f.coverage(), 'density' : f.density(), 'compression' : f.compression()})
		pickle.dump(data, open(pickle_f, 'wb'))
	print("Fetched CNN/DM Dataset with {} examples".format(len(data)))
	return data	