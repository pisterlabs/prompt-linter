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
nltk.download("stopwords")
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


def add_id(data):
    for i, d in enumerate(data):
        d["id"] = i


def add_sent_comp(data):
    for d in data:
        d["compression_sent"] = len(sent_tokenize(d["text"])) / len(
            sent_tokenize(d["summary"])
        )


def compute_compression_sent(data):
    return 1 - (sum([1 / ex["compression_sent"] for ex in data]) / len(data))


def add_abs1(data):
    print("Computing Abstractivity-1")
    for x in data:
        x["abs1"] = 1 - x["coverage"] / len(x)


def add_abs2(data, dataset):
    print("Computing Abstractivity-2")
    output = []
    for ex in tqdm(data):
        if dataset == "cnndm":
            ex["abs2"] = 1 - ex["density"] / len([w for w in ex["summary"].split()])
        else:
            ex["abs2"] = 1 - ex["density"] / len(spacy_tokenizer(ex["summary"]))


def add_red(data):
    print("Computing Redundancy")
    red1_output, red2_output, redL_output = [], [], []
    for ex in tqdm(data):
        summary = ex["summary"]
        red1_scores, red2_scores, redL_scores = [], [], []
        sentences = sent_tokenize(summary)
        sentences = [
            [str(token).lower() for token in spacy_tokenizer(s)] for s in sentences
        ]
        if len(sentences) <= 1:
            ex["red1"] = 0
            ex["red2"] = 0
            ex["redL"] = 0
        else:
            for i in range(len(sentences)):
                for j in range(
                    i + 1, len(sentences)
                ):  # ROUGE is symmetric, so only do one of (a,b), (b,a)
                    red1_scores.append(
                        ROUGE_N(sentences[i], sentences[j], 1)[2]
                    )  # Rouge Triple of (p, r, f)
                    red2_scores.append(ROUGE_N(sentences[i], sentences[j], 2)[2])
                    redL_scores.append(ROUGE_L(sentences[i], sentences[j])[2])
            ex["red1"] = max(red1_scores)
            ex["red2"] = max(red2_scores)
            ex["redL"] = max(redL_scores)


def add_sc(data):
    print("Computing Semantic Coherence")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForNextSentencePrediction.from_pretrained("bert-base-uncased")
    softmax = torch.nn.Softmax(dim=1)
    model.eval()
    output = []
    for ex in tqdm(data):
        summary = ex["summary"]
        scores = []
        sentences = sent_tokenize(summary)
        if len(sentences) <= 1:
            ex["coherence"] = 1
        else:
            numerator = 0
            denominator = len(sentences) - 1
            for i in range(len(sentences) - 1):
                prev = sentences[i]
                curr = sentences[i + 1]
                s = "[CLS] " + prev + " [SEP] " + curr + " [SEP]"
                tokenized_text = tokenizer.tokenize(s)
                boundary = tokenized_text.index("[SEP]")
                segment_ids = [0] * boundary + [1] * (len(tokenized_text) - boundary)
                indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
                tokens_tensor = torch.tensor([indexed_tokens])
                segments_tensors = torch.tensor([segment_ids])
                with torch.no_grad():
                    prediction = model(tokens_tensor, token_type_ids=segments_tensors)[
                        0
                    ]
                prediction_sm = softmax(prediction)[0].tolist()
                if prediction_sm[0] > 0.5:
                    numerator += 1
            ex["coherence"] = numerator / denominator


def add_all(data, dataset_name):
    """
  return cleaned data with each datapoint d in the following format
  d['summary']: String, summary text
  d['text']: String, full text
  d['coverage']: Abstractivity score
  d['density']
  d['compression']: Word compression (D/S)
  d['compression_sent']: Sentence compression (D/S)
  d['id']: index of data in dataset (0 index)
  d['abs1']: abstractivity 1 score 
  d['abs2']: abstractivity 2 score 
  d['coherence']: semantic coherence score
  d['coherence_sent']: semantic coherence score
  d['red1']: redundancy 1 score
  d['red2']: redundancy 2 score
  d['redL']: redundancy L score
  """
    add_id(data)
    add_red(data)
    add_abs1(data)
    add_abs2(data, dataset_name)
    add_sc(data)
    add_sent_comp(data)
    return data


def sort_by(t, data):
    """
  sorts in ascending order by key t(field in data)
  """
    return sorted(data, key=lambda x: x[t], reverse=False)


if __name__ == "__main__":
    print("todo")
    # open dataset
    # call add_all
    # returns cleaned data
