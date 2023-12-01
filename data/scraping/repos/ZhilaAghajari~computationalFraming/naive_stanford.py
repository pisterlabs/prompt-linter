# [TODO] create conda virtual environment for package management
import os
# os.chdir(os.environ['PROJECT_DIR'])

import pandas as pd
import numpy as np

from tqdm import tqdm
from p_tqdm import p_map # multiprocessing tqdm
from multiprocessing import Pool

import matplotlib.pyplot as plt
from graphviz import Source # graphing dependency tree

import stanza # stanford corenlp: https://stanfordnlp.github.io/stanza/
try:
    corenlp = stanza.Pipeline('en', processors="tokenize,mwt,pos,lemma,depparse", verbose=False, use_gpu=False)
except:
    stanza.download('en') # download corenlp neural model
    corenlp = stanza.Pipeline('en', processors="tokenize,mwt,pos,lemma,depparse", verbose=False, use_gpu=False)

# from datasets import load_dataset # hugging face datasets

import gensim.corpora as corpora
from gensim import utils
from gensim.models.coherencemodel import CoherenceModel

import nltk
from nltk import tokenize
try:
    nltk.corpus.stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

from lda.LDAMallet import LdaMallet # gensim LDA (gibbs sampling) mallet wrapper 

# use nltk stopwords
STOP_WORDS = nltk.corpus.stopwords.words('english')

def _get_word_reln_pairs(doc):
    '''Loop through sentences and words in sentences to get dependency relationships'''
    
    # add gov or dep based on head
    word_reln_pairs = [
        (word.text, word.deprel) 
        for sent in doc.sentences 
        for word in sent.words
    ]
    
    return word_reln_pairs

def _process_pairs(pairs, stop_words = STOP_WORDS):# -> list[tuple]:
    '''Remove words that are stop words, non-alphabetic, and less than 3 characters long'''
    
    valid_word = lambda word: not word in stop_words and word.isalpha() and len(word) > 2
    
    processed_pairs = []
    for word, reln in pairs:
        # [TODO] unicode and lower full documents instead of each word for increase efficiency
        processed_word = utils.to_unicode(word.lower())
        # if valid_word(processed_word):
        #     processed_pairs.append((processed_word, reln))
    
    return processed_pairs

def _concatenate_pairs(pairs, sep= "%"):# -> list[str]:
    '''Join dependency relational pairs into single strings using seperator'''
    
    join_tuple = lambda pair: sep.join(pair)
    strs = list(map(join_tuple, pairs))
    
    return strs

def coherence_optimization(tokens, id2word, corpus, topics_range):# -> tuple[list[any], list[any]]:
    '''
    Description
        Perform coherence optimization on LDA Mallet model. This finds the model that has the best coherence
        in relation to number of topics. Essentially finds the best number of topics for a given corpus.
    
    Params
        tokens: tokenized documents
        id2word: a Gensim dictionary mapping of id to word.
        corpus: list of documents in bag of word (BoW) format
    
    Returns
        model_list -> list[LdaMallet]
        coherence_values -> list[float]
    '''

    model_list, coherence_values = [], []
    for n_topics in tqdm(topics_range):
        model = LdaMallet(os.environ['MALLET_DIR'], corpus=corpus, num_topics=n_topics, id2word=id2word)
        coherence_model = CoherenceModel(model=model, texts=tokens, coherence='c_npmi')

        model_list.append(model)
        coherence_values.append(coherence_model.get_coherence())
    
    return model_list, coherence_values

def get_tokens(text):# str) -> list[str]:
    '''Obtain concatenated dependency relational pairs of text'''

    doc = corenlp(text)
    word_reln_pairs = _get_word_reln_pairs(doc)
    processed_pairs = _process_pairs(word_reln_pairs)
    word_reln_strs = _concatenate_pairs(processed_pairs, sep="?")
    
    return word_reln_strs

def get_topics(model: any, n_topics: int) -> dict:
    '''Returns dictionary of topics'''
    
    topics_dict = dict(model.print_topics(num_topics=n_topics))
    topics_dict = {int(k):v for k,v in topics_dict.items()}
    
    return topics_dict

def get_doc_top_matrix(model, n_topics):# any, n_topics: int) -> list[any]:
    '''Sort document topic matrix and add probability of 0 for topics that aren't included in documents'''

    doc_top_matrix = [*model.load_document_topics()]

    new_doc_top_matrix = []
    for doc_top in doc_top_matrix:
        _dict = dict(doc_top)
        for key in range(n_topics):
            if key not in _dict:
                _dict[key] = 0
        new_doc_top_matrix.append(list(_dict.items()))

    doc_top_matrix = [sorted(arr) for arr in new_doc_top_matrix]
    
    return doc_top_matrix

# [NOTE]: No longer needed
# def main():
#     # load dataset
#     # [TODO]: use scraped dataset
#     dailymail = load_dataset('cnn_dailymail', '2.0.0') # https://huggingface.co/datasets/cnn_dailymail/viewer/2.0.0/
#     texts = dailymail['train']['article'][:2000]

#     # process corpus 
#     tokens = p_map(get_tokens, texts)

#     # run Mallet LDA model on relational pairs and perform coherence optimization
#     id2word = corpora.Dictionary(tokens)
#     corpus = list(map(lambda x: id2word.doc2bow(x), tokens))
#     model_list, coherence_values = coherence_optimization(tokens, id2word, corpus, range(5, 31, 5))

#     # use highest coherence value for main model
#     model = model_list[np.argmax(coherence_values)]

#     # get document topic matrix
#     doc_top_matrix = get_doc_top_matrix(model, 5)

#     with open('model/naive_model.pkl', 'wb') as f:
#         pickle.dump(model, f)