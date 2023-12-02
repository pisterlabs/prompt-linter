# extra functions for jargon constraints

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import  word_tokenize
import numpy as np
import csv as csv
import pandas as pd
import string
import re
from collections import Counter
import spacy
import requests
from transformers import OpenAIGPTTokenizer, OpenAIGPTLMHeadModel
import math
from tqdm import tqdm
import torch

import os

SCI_ARTICLES_DATA_DIR = ''
SCI_ARTICLES_RESOURCE_DIRECTORY = '{}/resources'.format(SCI_ARTICLES_DATA_DIR)
SPACY_EN = spacy.load("en_core_web_sm")

wnl = WordNetLemmatizer()

def lemmatize(wnl, tokens):
    return [wnl.lemmatize(t) for t in tokens]

def num_words(tokens, word_list, return_words=False):
    counts = Counter(tokens)
    word_counts = {k: counts[k] for k in counts.keys() & set(word_list)}
    if return_words:
        return word_counts
    return sum(word_counts.values())

def jargon_normalized(passage, jargon_words):
    tokens = word_tokenize(passage)
    text_len = len(tokens)
    token_lemmas = lemmatize(wnl, tokens)
    
    return num_words(token_lemmas, jargon_words)/text_len


def get_jargon_lists():
	df_word_list_dejargonizer = pd.read_csv(SCI_ARTICLES_RESOURCE_DIRECTORY+'/jargon_word_list.csv')

	AVL_core = pd.read_excel(SCI_ARTICLES_RESOURCE_DIRECTORY+'/acadCore.xlsx', sheet_name=1) 

	jargon_words = df_word_list_dejargonizer['General science jargon']
	jargon_words = jargon_words.append(df_word_list_dejargonizer['Science common words'])
	jargon_words = jargon_words.append(AVL_core['word'])

	return jargon_words


# using spacy now, and masking any word that falls in the jargon list
def mask_jargon(passage, jargon_words, nlp_spacy=SPACY_EN):
    doc = nlp_spacy(passage)
    masked_doc = []
    for token in doc:
        if token.lemma_ in list(jargon_words):
            masked_doc.append('<'+token.pos_+'>')
        else:
            masked_doc.append(token.text)
    return masked_doc


#### loading
with open('{}/full_avl_set.txt'.format(JARGON_RESOURCE_DIRECTORY), 'r') as f:
    AVL_set = f.readlines()
    
AVL_set = set([t.strip('\n') for t in AVL_set])

def get_avl_occ(text, AVL_core_lemmas_set=AVL_set):
    lemmas = [t.lemma_ for t in text]
    return sum(x in AVL_core_lemmas_set for x in lemmas)/len(lemmas)


################################
### Thing Explainer
################################
response = requests.get('https://splasho.com/upgoer5/phpspellcheck/dictionaries/1000.dicin')
top_1000 = response.text.split('\n')[:-1] # last one is TRUE?

def get_thing_explainer_oov(r, top_1000):
    # lemmatize
    lemmas = [t.lemma_ for t in r]
    
    # get occurances tokens out of top 1000
    return sum(x not in set(top_1000) for x in lemmas)/len(lemmas)


################################
### Readability
################################
# do it for the full dataset
def get_readability(r):
    try:
        return Readability(r).flesch_kincaid().score
    except:
        return None

################################
### Function Words
################################
funct_pos_tags = ['DET', 'ADP', 'PRON', 'CONJ', 'SCONJ', 'AUX', 'PART', 'INTJ']

def get_num_function_words(text, funct_pos_tags):
    counts = Counter([t.pos_ for t in text])
    return sum([counts[k] for k in funct_pos_tags])


################################
### GPT Perplexity
################################


tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
model = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt')
_ = model.eval()

# from: https://github.com/huggingface/transformers/issues/473
def score(sentence, tokenizer, model):
    # tokenize_input = tokenizer.tokenize(sentence, truncate)
    # tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
    tensor_input = torch.tensor([tokenizer(sentence, truncation=True)['input_ids']])

    outputs=model(tensor_input, labels=tensor_input)
    return math.exp(outputs[0])


##### Make the features -- assumes many variables from the cell above, so don't move
def make_jargon_features(df, text_col='sentence'):
    # tokenize first - check that it doesn't exist already because it is a hassle to do over
    if 'sent_tokens' not in df.columns:
        print('Tokenizing....', end='')
        df['sent_tokens'] = [SPACY_EN(s) for s in tqdm(df[text_col])]
        print('Done')
    
    # Word count
    df['word_count'] = [len(s) for s in df['sent_tokens']]

    # AVL
    df['avl_occ'] = [get_avl_occ(t) for t in df['sent_tokens']]

    # Thing Explainer
    df['te_oov'] = [get_thing_explainer_oov(r, top_1000) for r in df['sent_tokens']]

    # Function Words
    df['function_words'] = [get_num_function_words(t, funct_pos_tags) for t in df['sent_tokens']]
    df['function_words_prop'] = df['function_words']/df['word_count']

    if 'response_gpt_ppl_score' not in df.columns:
        # GPT Perplexity
        print('Getting GPT perplexity....', end='')
        df['response_gpt_ppl_score'] = [score(s, tokenizer, model) for s in tqdm(df[text_col])]
        print('Done')
    return df.copy()
    
