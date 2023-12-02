import pandas as pd
import numpy as np
import pickle
import re

import spacy

import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel, HdpModel
from nltk.corpus import stopwords

# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
nlp = spacy.load('en', disable=['parser', 'ner'])


def sent_to_words(sentences):
    for sentence in sentences:
        # deacc=True removes punctuations
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))


def remove_stopwords(texts, stop_words):
    return [[word for word in gensim.utils.simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]


def make_bigrams(texts, bigram_mod):
    return [bigram_mod[doc] for doc in texts]


def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append(
            [token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


def create_dictionary(data_lemmatized):
    # Create Dictionary
    id2word = corpora.Dictionary(data_lemmatized)
    # Filter words
    id2word.filter_extremes(no_below=5, no_above=0.95,
                            keep_n=1800, keep_tokens=None)
    return id2word


def create_model_corpus(id2word, data_lemmatized):
    return [id2word.doc2bow(text) for text in data_lemmatized]


def build_hdp(corpus, id2word):
    hdpmodel = HdpModel(corpus=corpus, id2word=id2word, chunksize=2000)
    hdptopics = hdpmodel.show_topics(formatted=False)
    hdptopics = [[word for word, prob in topic]
                 for topicid, topic in hdptopics]
    return hdpmodel, hdptopics


def calc_confusion_matrix(model1, model2, corpus1, doc_max=True):
    lda_corpus_1 = [max(prob, key=lambda y:y[1])
                    for prob in model1[corpus1]]

    lda_corpus_2 = [max(prob, key=lambda y:y[1])
                    for prob in model2[corpus1]]
    # print(lda_corpus_1 == lda_corpus_2)

#     print('Corpus 1 Length - ', len(lda_corpus_1))
    # print('Corpus 2 Length - ', len(lda_corpus_2))
    # max_corpus_size = max(len(lda_corpus_1), len(lda_corpus_2))
    # min_corpus_size = min(len(lda_corpus_1), len(lda_corpus_2))
    # print(max_corpus_size)

    positive = 0
    negative = 0

    # if(doc_max == True):
    #     upper_limit = max_corpus_size
    # else:
    #     upper_limit = min_corpus_size
    upper_limit = len(lda_corpus_1)
    total_permutations = upper_limit * \
        (upper_limit-1)/2  # nC2 combinations

    for i in range(upper_limit):
        for j in range(i+1, upper_limit):
            # print(upper_limit)
            # print(i)
            # print(j)
            if(lda_corpus_1[i][0] == lda_corpus_1[j][0] and lda_corpus_2[i][0] == lda_corpus_2[j][0]):
                positive = positive+1
            elif(lda_corpus_1[i][0] != lda_corpus_1[j][0] and lda_corpus_2[i][0] == lda_corpus_2[j][0]):
                negative = negative+1
            elif(lda_corpus_1[i][0] == lda_corpus_1[j][0] and lda_corpus_2[i][0] != lda_corpus_2[j][0]):
                negative = negative+1
            elif(lda_corpus_1[i][0] != lda_corpus_1[j][0] and lda_corpus_2[i][0] != lda_corpus_2[j][0]):
                positive = positive+1

    answers_positive = (round(positive*100/total_permutations, 2))
    # answers_negative = (round(negative*100/total_permutations, 2))

    return answers_positive
