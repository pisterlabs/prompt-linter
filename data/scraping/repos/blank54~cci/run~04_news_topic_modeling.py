'''
https://www.machinelearningplus.com/nlp/topic-modeling-python-sklearn-examples/
https://coredottoday.github.io/2018/09/17/%EB%AA%A8%EB%8D%B8-%ED%8C%8C%EB%9D%BC%EB%AF%B8%ED%84%B0-%ED%8A%9C%EB%8B%9D/
'''

#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os
import sys
rootpath = os.path.sep.join(os.path.dirname(os.path.abspath(__file__)).split(os.path.sep)[:-1])
sys.path.append(rootpath)

from news import NewsCorpus
from newsutil import NewsIO, NewsPath
newsio = NewsIO()
newspath = NewsPath()

import json
import itertools
import numpy as np
from tqdm import tqdm

import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel


def data_preparation(corpus, SAMPLE_SIZE):
    ## Filenames
    fname_docs_dict = f'lda/docs_dict_{SAMPLE_SIZE}.json'
    fname_id2word = f'lda/id2word_{SAMPLE_SIZE}.json'
    fname_docs_bow = f'lda/docs_bow_{SAMPLE_SIZE}.json'

    try:
        docs_dict = newsio.load(_type='data', fname_object=fname_docs_dict)
        id2word = newsio.load(_type='data', fname_object=fname_id2word)
        docs_bow = newsio.load(_type='data', fname_object=fname_docs_bow)
    except FileNotFoundError:
        docs_dict = {}
        for article in corpus.iter_sampling(n=SAMPLE_SIZE):
            docs_dict[article.id] = list(itertools.chain(*article.nouns_stop))

        id2word = corpora.Dictionary(docs_dict.values())
        docs_bow = [id2word.doc2bow(text) for text in docs_dict.values()]

        docs_dict = newsio.save(_object=docs_dict, _type='data', fname_object=fname_docs_dict)
        id2word = newsio.save(_object=id2word, _type='data', fname_object=fname_id2word)
        docs_bow = newsio.save(_object=docs_bow, _type='data', fname_object=fname_docs_bow)

    return docs_dict, id2word, docs_bow

def develop_lda_model(docs_bow, id2word, num_topics, iterations, alpha, eta):
    lda_model = LdaModel(corpus=docs_bow,
                         id2word=id2word,
                         num_topics=num_topics,
                         iterations=iterations,
                         alpha=alpha,
                         eta=eta,)

    return lda_model

def calculate_coherence(lda_model, docs_dict):
    coherence_model = CoherenceModel(model=lda_model,
                                     texts=docs_dict.values())

    coherence_score = coherence_model.get_coherence()
    if np.isnan(coherence_score):
        return 0
    else:
        return coherence_score

def gridsearch(fname_gs_result, do, **kwargs):
    if do:
        docs_dict = kwargs.get('docs_dict')
        id2word = kwargs.get('id2word')
        docs_bow = kwargs.get('docs_bow')

        parameters = kwargs.get('parameters')
        num_topics_list = parameters.get('num_topics')
        iterations_list = parameters.get('iterations')
        alpha_list = parameters.get('alpha')
        eta_list = parameters.get('eta')

        gs_result = {}
        candidates = itertools.product(*[num_topics_list, iterations_list, alpha_list, eta_list])
        for candidate in tqdm(candidates):
            print('\n--------------------------------------------------')
            print(f'LDA modeling')
            print(f'  | candidate: {candidate}')
            num_topics, iterations, alpha, eta = candidate
            fname_lda_model = f'lda/lda_{len(docs_dict)}_{num_topics}_{iterations}_{alpha}_{eta}.pk'
            fname_coherence = f'lda/coherence_{len(docs_dict)}_{num_topics}_{iterations}_{alpha}_{eta}.pk'

            try:
                lda_model = newsio.load(_type='model', fname_object=fname_lda_model)
            except FileNotFoundError:
                lda_model = develop_lda_model(docs_bow, id2word, num_topics, iterations, alpha, eta)
                newsio.save(_object=lda_model, _type='model', fname_object=fname_lda_model)

            try:
                coherence_score = newsio.load(_type='model', fname_object=fname_coherence)
            except FileNotFoundError:
                coherence_score = calculate_coherence(lda_model, docs_dict)
                newsio.save(_object=coherence_score, _type='model', fname_object=fname_coherence)
            gs_result[fname_lda_model] = coherence_score
            
            print('--------------------------------------------------')
            print(f'LDA result')
            print(f'  | candidate: {candidate}')
            print(f'  | coherence: {coherence_score:,.03f}')

        newsio.save(_object=gs_result, _type='result', fname_object=fname_gs_result)

    else:
        gs_result = newsio.load(_type='result', fname_object=fname_gs_result)

    return gs_result


if __name__ == '__main__':
    ## Filenames

    ## Parameters
    SAMPLE_SIZE = 100000

    DO_DATA_PREPARATION = True
    DO_GRIDSEARCH = True

    GS_PARAMETERS = {'num_topics': list(range(3, 51, 1)),
                     'iterations': [10, 50, 100, 500],
                     'alpha': [0.01, 0.02, 0.05, 0.1],
                     'eta': [0.01, 0.02, 0.05, 0.1],
                    }

    ## Data import
    print('============================================================')
    print('--------------------------------------------------')
    print('Load corpus')

    corpus = NewsCorpus()
    docs_dict, id2word, docs_bow = data_preparation(corpus=corpus, SAMPLE_SIZE=SAMPLE_SIZE)
    print(f'  | Corpus     : {len(corpus):,}')
    print(f'  | Sample size: {SAMPLE_SIZE:,}')

    ## Filenames
    fname_gs_result = f'lda_gs_{SAMPLE_SIZE}.json'

    ## Topic modeling
    print('============================================================')
    print('Gridsearch')

    gs_result = gridsearch(fname_gs_result=fname_gs_result,
                           docs_dict=docs_dict,
                           id2word=id2word,
                           docs_bow=docs_bow,
                           do=DO_GRIDSEARCH,
                           parameters=GS_PARAMETERS)
    print(gs_result)