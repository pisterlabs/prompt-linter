import random
import torch
import math
import jsonlines as jsonl
from fragments import Fragments
from rouge.rouge import rouge_n_sentence_level as ROUGE_N
from rouge.rouge import rouge_l_sentence_level as ROUGE_L
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
from fetch_data import * 
from converter import *
from topic_similarity import *


def compute_distributional_statistics(data):
	output = {'sum_word_length' : 0, 'sum_sent_length' : 0, 'doc_word_length' : 0, 'doc_sent_length' : 0}
	for example in tqdm(data):
		s = example['summary']
		d = example['text']
		output['sum_sent_length'] += len(sent_tokenize(s))
		output['doc_sent_length'] += len(sent_tokenize(d))
		output['sum_word_length'] += len(nlp(s))
		output['doc_word_length'] += len(nlp(d))
	output = {k : v / len(data) for (k,v) in output.items()}
	return output


def load_datasets():
	N = 20000
	our_datasets = {'cnndm' : fetch_cnndm, 'nyt' : fetch_nyt , 'newsroom' : fetch_newsroom, 'tldr' : fetch_tldr, 'gigaword' : fetch_gigaword}
	for dataset_name, fetcher in our_datasets.items():
		data = fetcher(pickled = False, N = N)
		print(len(data))
		data = data[:N]
		our_statistics = compute_our_statistics(data, dataset_name)
	# 	# cmu_version = us2cmu(data)
	# 	# data = []
	# 	# cmu_statistics = compute_cmu_statistics(cmu_version, dataset_name)
	# 	# cmu_version = []
	# 	print(compute_distributional_statistics(data))

	# cmu_datasets = fetch_cmu({'ami' : None, 'moviescript' : None, 'peerread' : None, 'pubmed' : None, 'xsum' : None})
	# for dataset_name, cmu_version in cmu_datasets.items():
	# 	#cmu_statistics = compute_cmu_statistics(cmu_version, dataset_name)
	# 	data = cmu2us(cmu_version)
	# 	cmu_version = []
	# 	data = data[:N]
	# 	print(dataset_name)
	# 	# print(compute_distributional_statistics(data))
	# 	our_statistics = compute_our_statistics(data, dataset_name)
		# data = []
		# print(dataset_name, our_statistics)


def compute_word_compression(data):
	print("Computing Word Compression")
	return 1 - (sum([1 / ex['compression'] for ex in data]) / len(data))


def compute_sentence_compression(data):
	print("Computing Sentence Compression")
	return 1 - (sum([len(sent_tokenize(ex['summary']))/len(sent_tokenize(ex['text'])) for ex in data]) / len(data))


def compute_abs1(data):
	print("Computing Abstractivity-1")
	return 1 - (sum([ex['coverage'] for ex in data]) / len(data))


def compute_abs2(data, dataset):
	print("Computing Abstractivity-2")
	output = []
	for ex in tqdm(data):
		if dataset == 'cnndm':
			output.append(ex['density'] / len([w for w in ex['summary'].split()]))
		else:
			output.append(ex['density'] / len(spacy_tokenizer(ex['summary'])))
	# return 1 - output/len(output)
	return 1 - (sum(output) / len(output))


def compute_red(data):
	print("Computing Redundancy")
	red1_output, red2_output, redL_output = [], [], []
	for ex in tqdm(data):
		summary = ex['summary']
		red1_scores, red2_scores, redL_scores = [], [], []
		sentences = sent_tokenize(summary)
		sentences = [[str(token).lower() for token in spacy_tokenizer(s)] for s in sentences]
		if len(sentences) <= 1:
			red1_output.append(0)
			red2_output.append(0)
			redL_output.append(0)
		else:
			for i in range(len(sentences)):
				for j in range(i + 1, len(sentences)): # ROUGE is symmetric, so only do one of (a,b), (b,a)
					red1_scores.append(ROUGE_N(sentences[i], sentences[j], 1)[2]) # Rouge Triple of (p, r, f)
					red2_scores.append(ROUGE_N(sentences[i], sentences[j], 2)[2])
					redL_scores.append(ROUGE_L(sentences[i], sentences[j])[2])
			red1_output.append(max(red1_scores))
			red2_output.append(max(red2_scores))
			redL_output.append(max(redL_scores))
	assert len(red1_output) == len(data)
	assert len(red2_output) == len(data)
	assert len(redL_output) == len(data)
	return sum(red1_output) / len(red1_output), sum(red2_output) / len(red2_output), sum(redL_output) / len(redL_output)

# zip output with data and sort 
def compute_sc(data):
	print("Computing Semantic Coherence")
	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
	model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')
	softmax = torch.nn.Softmax(dim=1)
	model.eval()
	output = []
	for ex in tqdm(data):
		summary = ex['summary']
		scores = []
		sentences = sent_tokenize(summary)
		if len(sentences) <= 1:
			output.append(1)
		else:
			numerator = 0
			denominator = len(sentences) - 1
			for i in range(len(sentences) - 1):
				prev = sentences[i]
				curr = sentences[i + 1]
				s = "[CLS] " + prev + " [SEP] " + curr + " [SEP]"
				tokenized_text = tokenizer.tokenize(s)
				boundary = tokenized_text.index('[SEP]')
				segment_ids = [0] * boundary + [1] * (len(tokenized_text) - boundary)
				indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
				tokens_tensor = torch.tensor([indexed_tokens])
				segments_tensors = torch.tensor([segment_ids])
				with torch.no_grad():
					prediction = model(tokens_tensor, token_type_ids=segments_tensors)[0]
				prediction_sm = softmax(prediction)[0].tolist()
				if prediction_sm[0] > 0.5:
					numerator += 1
			output.append(numerator / denominator)
	print(len(data), len(output))
	# assert len(output) == len(data)
	return sum(output) / len(output)


def compute_our_statistics(data, dataset):
	print("Computing statistics for:", dataset)
	word_compression, sentence_compression,  abs1, abs2, red1, red2, redL, semantic_coherence = [None] * 8
	topic_similarity = {(k, input_data) : 0 for k in [10, 25, 50, 100] for input_data in ['document', 'joint']}

	# word_compression = compute_word_compression(data)
	# sentence_compression = compute_sentence_compression(data)
	for k, input_data in topic_similarity:
		if input_data == 'document':
			topic_similarity[k, input_data] = compute_ts_document(data, dataset, k, pickled_corpus=False, pickled_model=False)
		elif input_data == 'joint':
			topic_similarity[k, input_data] = compute_ts_joint(data, dataset, k, pickled_corpus=False, pickled_model=False)
	# abs1 = compute_abs1(data)
	# abs2 = compute_abs2(data, dataset)
	# red1, red2, redL = compute_red(data)
	# print({"CMP_W" : word_compression, "CMP_S" : sentence_compression, "TS" : topic_similarity, "ABS1" : abs1, "ABS2" : abs2, "RED1" : red1, "RED2" : red2, "REDL" : redL, "SC" : semantic_coherence})
	# semantic_coherence = compute_sc(data)
	print({"CMP_W" : word_compression, "CMP_S" : sentence_compression, "TS" : topic_similarity, "ABS1" : abs1, "ABS2" : abs2, "RED1" : red1, "RED2" : red2, "REDL" : redL, "SC" : semantic_coherence})
	# print()
	# print()
	# print()
	return {"CMP_W" : word_compression, "CMP_S" : sentence_compression, "TS" : topic_similarity, "ABS1" : abs1, "ABS2" : abs2, "RED1" : red1, "RED2" : red2, "REDL" : redL, "SC" : semantic_coherence}
	

if __name__ == '__main__':
	load_datasets()
	# cnndm_data = fetch_cnndm(pickled=True)
	# cnndm_data = cnndm_data[:20000]
	# newsroom_data = fetch_newsroom(pickled=True)
	# newsroom_data = newsroom_data[:20000]

	# tldr_data = fetch_tldr(pickled=False)
	# tldr_data = tldr_data[:20000]

	# note: can run add_sent_comp on data or just add it to when reading data
	# example: add_sent_comp(tldr_data)

	# gigaword_data = fetch_gigaword(pickled=True)
	# gigaword_data = gigaword_data[:20000]

	# nyt_data = fetch_nyt(pickled=True)
	
	# nyt_data = nyt_data[:20000]

	# exit()
	# statistics = compute_statistics(nyt_data, 'nyt')
	# print(statistics)``

	# return
	# statistics = compute_statistics(cnndm_data, 'cnndm')
	# print(statistics)
	# statistics = compute_statistics(newsroom_data, 'newsroom')
	# print(statistics)
	# statistics = compute_statistics(tldr_data, 'tldr')
	# print(statistics)
	# statistics = compute_statistics(gigaword_data, 'gigaword')
	# print(statistics)
	# return
	