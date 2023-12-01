#!/usr/bin/env python -W ignore::DeprecationWarning
# -*- coding: utf-8 -*-

# Configuration
import os
import sys
import random
import json
import psutil
import itertools
import numpy as np
import pickle as pk
import pandas as pd
from tqdm import tqdm
from glob import glob
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import gensim.corpora as corpora
from gensim.models.ldamodel import LdaModel
from gensim.models import CoherenceModel
import pyLDAvis.gensim

import matplotlib.pyplot as plt


class NewsPath:
    root = os.path.dirname(os.path.abspath(__file__))

    fdir_data = os.path.sep.join((root, 'data'))
    
    fdir_corpus = os.path.sep.join((root, 'corpus'))
    fdir_model = os.path.sep.join((root, 'model'))
    fdir_thesaurus = os.path.sep.join((root, 'thesaurus'))
    fdir_result = os.path.sep.join((root, 'result'))

    fdir_query = os.path.sep.join((root, 'query'))
    fdir_url_list = os.path.sep.join((fdir_data, 'url_list'))
    fdir_articles = os.path.sep.join((root, 'articles'))


class NewsIO(NewsPath):
    def memory_usage(self):
        print('------------------------------------------------------------')
        active_memory = psutil.virtual_memory()._asdict()['used']
        print('  | Current memory usage: {:,.03f} GB ({:,.03f} MB)'.format(active_memory/(2**30), active_memory/(2**20)))
        print('--------------------------------------------------')

    def save(self, _object, fname_object, verbose=True, **kwargs):
        _type = kwargs.get('_type', '')
        fdir_object = kwargs.get('fdir_object', os.path.sep.join((self.root, _type)))
        fpath_object = os.path.sep.join((fdir_object, fname_object))

        os.makedirs(fdir_object, exist_ok=True)
        with open(fpath_object, 'wb') as f:
            pk.dump(_object, f)

        if verbose:
            print(f'  | fdir : {fdir_object}')
            print(f'  | fname: {fname_object}')

    def save_json(self, _object, _type, fname_object, verbose=True):
        fdir_object = os.path.sep.join((self.root, _type))
        fpath_object = os.path.sep.join((fdir_object, fname_object))

        os.makedirs(fdir_object, exist_ok=True)
        with open(fpath_object, 'w', encoding='utf-8') as f:
            json.dump(_object, f)

        if verbose:
            print(f'  | fdir : {fdir_object}')
            print(f'  | fname: {fname_object}')

    def load(self, fname_object, verbose=True, **kwargs):
        _type = kwargs.get('_type', '')
        fdir_object = kwargs.get('fdir_object', os.path.sep.join((self.root, _type)))
        fpath_object = os.path.sep.join((fdir_object, fname_object))
        with open(fpath_object, 'rb') as f:
            _object = pk.load(f)

        if verbose:
            print(f'  | fdir : {fdir_object}')
            print(f'  | fname: {fname_object}')

        return _object

    def load_json(self, fname_object, _type, verbose=True):
        fdir_object = os.path.sep.join((self.root, _type))
        fpath_object = os.path.sep.join((fdir_object, fname_object))
        with open(fpath_object, 'r', encoding='utf-8') as f:
            _object = json.load(f)

        if verbose:
            print(f'  | fdir : {fdir_object}')
            print(f'  | fname: {fname_object}')

        return _object

    def load_cci(self, start, end):
        fpath = os.path.sep.join((self.fdir_data, 'cci.xlsx'))
        df = pd.read_excel(fpath, na_values='')

        data = defaultdict(list)
        for _, row in df.iterrows():
            year = datetime.strftime(row['yearmonth'], '%Y')
            month = datetime.strftime(row['yearmonth'], '%m')
            yearmonth = '{}{}'.format(year, month)
            if yearmonth < start:
                continue
            elif yearmonth > end:
                continue
            else:
                pass

            if row['cci']:
                data['yearmonth'].append(yearmonth)
                data['cci'].append(row['cci'])
            else:
                print(f'Error: no value of CCI at {yearmonth}')
                continue

        return pd.DataFrame(data)

    def read_thesaurus(self, fname_thesaurus):
        fpath_thesaurus = os.path.sep.join((self.fdir_thesaurus, fname_thesaurus))
        with open(fpath_thesaurus, 'r', encoding='utf-8') as f:
            word_list = list(set([w.strip() for w in f.read().strip().split('\n')]))

        return word_list


class NewsFunc(NewsPath):
    def text2sents(self, text):
        '''
        text : a str object of doc.content
        '''

        sents = ['{}다.'.format(sent) for sent in text.split('다.')]
        return sents

    def parse_fname_url_list(self, fname_url_list):
        query_part, date_part = fname_url_list.replace('.pk', '').split('_')
        query = query_part.split('-')[-1]
        date = date_part.split('-')[-1]
        return query, date

    def explore_demographic_info(self, df, **kwargs):
        except_list = kwargs.get('except_list', [])

        result = defaultdict(list)
        for attr in df.keys():
            if attr in except_list:
                continue
            else:
                pass
            
            try:
                max_value = max(df[attr])
                min_value = min(df[attr])
                
                if all([(max_value==0), (min_value==0)]):
                    print(f'Error: every element is zero: {attr}')
                    continue
                else:
                    result['attr'].append(attr)
                    result['max'].append(max_value)
                    result['min'].append(min_value)
                    result['mean'].append(df[attr].mean())
                    result['std'].append(df[attr].std())
            except TypeError:
                print(f'Error: wrong data type: {attr}')
                continue
            
        return pd.DataFrame(result)

    def partition_variable_list(self, demographic_info_df):
        variable_list_meanstd = []
        variable_list_minmax = []

        for _, row in demographic_info_df.iterrows():
            if row['min'] < 0:
                variable_list_meanstd.append(row['attr'])
            else:
                variable_list_minmax.append(row['attr'])

        return variable_list_meanstd, variable_list_minmax

    def normalize_meanstd(self, df):
        return (df-df.mean())/df.std()

    def normalize_minmax(self, df):
        return (df-df.min())/(df.max()-df.min())


# class NewsArticle:
#     '''
#     A class of news article.

#     Attributes
#     ----------
#     url : str
#         | The article url.
#     id : str
#         | The identification code for the article.
#     query : list
#         | A list of queries that were used to search the article.
#     title : str
#         | The title of the article.
#     date : str
#         | The uploaded date of the article. (format : yyyymmdd)
#     category : str
#         | The category that the article belongs to.
#     content : str
#         | The article content.
#     content_normalized : str
#         | Normalized content of the article.

#     Methods
#     -------
#     extend_query
#         | Extend the query list with the additional queries that were used to search the article.
#     '''

#     def __init__(self, **kwargs):
#         self.fpath_article = kwargs.get('fpath_article', '')
#         self.fpath_corpus = kwargs.get('fpath_corpus', '')

#         self.url = kwargs.get('url', '')
#         self.id = kwargs.get('id', '')
#         self.query = []

#         self.title = kwargs.get('title', '')
#         self.date = kwargs.get('date', '')
#         self.category = kwargs.get('category', '')
#         self.content = kwargs.get('content', '')

#         self.preprocess = False
#         self.sents = kwargs.get('sents', '')
#         self.normalized_sents = kwargs.get('normalized_sents', '')
#         self.nouns = kwargs.get('nouns', '')
#         self.nouns_stop = kwargs.get('nouns_stop', '')

#     def extend_query(self, query_list):
#         '''
#         A method to extend the query list.

#         Attributes
#         ----------
#         query_list : list
#             | A list of queries to be extended.
#         '''

#         queries = self.query
#         queries.extend(query_list)
#         self.query = list(set(queries))


class NewsDate:
    '''
    A class of news dates to address the encoding issues.

    Attributes
    ----------
    date : str
        | Date in string format. (format : yyyymmdd)
    formatted : datetime
        | Formatted date.
    '''

    def __init__(self, date):
        self.date = date
        self.yearmonth = self.__yearmonth()
        self.datetime = self.__datetime()

    def __call__(self):
        return self.datetime

    def __str__(self):
        return '{}'.format(self.__call__())

    def __datetime(self):
        try:
            return datetime.strptime(self.date, '%Y%m%d').strftime('%Y.%m.%d')
        except:
            return ''

    def __yearmonth(self):
        return datetime.strptime(self.date, '%Y%m%d').strftime('%Y%m')


class NewsCorpus():
    def __init__(self, **kwargs):
        self.dname_corpus = kwargs.get('dname_corpus', 'corpus')
        self.fdir_corpus = os.path.sep.join((NewsPath().root, self.dname_corpus))

        # self.yearmonth_list = sorted([dirs for _, dirs, _ in os.walk(os.path.sep.join((self.fdir_corpus, 'yearmonth'))) if dirs])
        self.start = kwargs.get('start', sorted(os.listdir(self.fdir_corpus))[0])
        self.end = kwargs.get('end', sorted(os.listdir(self.fdir_corpus))[-1])

        self.yearmonth_list = self.__get_yearmonth_list()
        self.fdir_list = [os.path.sep.join((self.fdir_corpus, yearmonth)) for yearmonth in self.yearmonth_list]

        self.topic_filtered = kwargs.get('topic_filtered', False)
        self.topic_ids = kwargs.get('topic_ids', []) # Topic ids to be filtered

    def __len__(self):
        corpus_len = 0
        if self.topic_filtered:
            for doc in self.iter():
                if doc['topic_id'] in self.topic_ids:
                    continue
                else:
                    corpus_len += 1
        else:
            for yearmonth in self.yearmonth_list:
                corpus_len += len(glob(self.fdir_corpus+'/'+yearmonth+'/*.json'))
        return corpus_len

    def sent_cnt(self):
        send_cnt = 0
        if self.topic_filtered:
            for doc in self.iter():
                if doc['topic_id'] in self.topic_ids:
                    continue
                else:
                    sent_cnt += len(doc['normalized_sents'])
        else:
            for doc in self.iter():
                send_cnt += len(doc['normalized_sents'])
        return sent_cnt

    def __get_yearmonth_list(self):
        yearmonth_start = datetime.strptime(self.start, '%Y%m').strftime('%Y-%m-%d')
        yearmonth_end = datetime.strptime(self.end, '%Y%m').strftime('%Y-%m-%d')
        return pd.date_range(yearmonth_start, yearmonth_end, freq='MS').strftime('%Y%m').tolist()

    def iter(self, sampling=False):
        fpath_list = list(itertools.chain(*[[os.path.sep.join((fdir, fname)) for fname in os.listdir(fdir)] for fdir in self.fdir_list]))

        if sampling:
            fpath_list = random.sample(fpath_list, k=n)
        else:
            pass

        if self.topic_filtered:
            for fpath in tqdm(fpath_list):
                try:
                    with open(fpath, 'r', encoding='utf-8') as f:
                        doc = json.load(f)
                        if doc['topic_id'] in self.topic_ids:
                            continue
                        else:
                            yield doc
                except:
                    print(f'ArticleReadingError: {fpath}')
                    yield None
        else:
            for fpath in tqdm(fpath_list):
                # try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    yield json.load(f)
                # except:
                #     print(f'ArticleReadingError: {fpath}')
                #     yield None

    def iter_month(self):
        if self.topic_filtered:
            for yearmonth in tqdm(self.yearmonth_list):
                doc_list = []
                for fpath in glob(self.fdir_corpus+'/'+yearmonth+'/*.json'):
                    try:
                        with open(fpath, 'r', encoding='utf-8') as f:
                            doc = json.load(f)
                            if doc['topic_id'] in self.topic_ids:
                                continue
                            else:
                                doc_list.append(doc)
                    except:
                        print(f'ArticleReadingError: {fpath}')
                yield yearmonth, doc_list
        else:
            for yearmonth in tqdm(self.yearmonth_list):
                doc_list = []
                for fpath in glob(self.fdir_corpus+'/'+yearmonth+'/*.json'):
                    try:
                        with open(fpath, 'r', encoding='utf-8') as f:
                            doc_list.append(json.load(f))
                    except:
                        print(f'ArticleReadingError: {fpath}')
                yield yearmonth, doc_list


class Word:
    def __init__(self, word):
        self.word = word

        self.tf = ''
        self.df = ''
        self.idf = ''
        self.tfidf = ''

    def __str__(self):
        return word


class LdaGridSearchResult:
    def __init__(self, gs_result, ignore_zero=True):
        self.fname2coherence = gs_result
        self.ignore_zero = ignore_zero
        self.result = self.__get_result()

    def __get_result(self):
        result = defaultdict(list)
        for fname, coherence in self.fname2coherence.items():
            if np.isnan(coherence):
                if self.ignore_zero:
                    continue
                else:
                    result['coherence'].append(0)
                    pass
            else:
                result['coherence'].append(coherence)

            _, doc_cnt, num_topics, iterations, alpha, eta = Path(fname).stem.split('_')
            result['fname'].append(fname)
            result['num_topics'].append(num_topics)
            result['iterations'].append(iterations)
            result['alpha'].append(alpha)
            result['eta'].append(eta)

        return result

    def scatter_plot(self, x):
        '''
        Attributes
        ----------
        x : str
            | The variable name (e.g., num_topics, iterations, alpha, eta)
        '''

        plt.scatter(self.result[x], self.result['coherence'])
        plt.show()

    def box_plot(self, x):
        '''
        Attributes
        ----------
        x : str
            | The variable name (e.g., num_topics, iterations, alpha, eta)
        '''

        _dict = defaultdict(list)
        for x_value, coherence in zip(self.result[x], self.result['coherence']):
            _dict[x_value].append(coherence)

        plt.boxplot(_dict.values())
        plt.xticks(range(1, len(_dict.keys())+1), _dict.keys())
        plt.show()


class NumericMeta:
    def __init__(self, fname, fdir):
        self.fname = fname
        self.fdir = fdir
        self.fpath = os.path.sep.join((self.fdir, self.fname))

        self.info = self.__read_metadata()
        self.kor2eng = {k: e for k, e, a in self.info}
        self.kor2abb = {k: a for k, e, a in self.info}
        self.eng2kor = {e: k for k, e, a in self.info}
        self.eng2abb = {e: a for k, e, a in self.info}
        self.abb2kor = {a: k for k, e, a in self.info}
        self.abb2eng = {a: e for k, e, a in self.info}

    def __read_metadata(self):
        with open(self.fpath, 'r', encoding='utf-8') as f:
            return [row.split('  ') for row in f.read().strip().split('\n')]

    def __len__(self):
        return len(self.info)

    def __str__(self):
        return self.info


class NumericData():
    def __init__(self, fdir, **kwargs):
        self.fdir = fdir

        self.data_list, self.var_list = self.__read_data()
        self.vars = list(set([var for _, var, _ in self.data_list]))

        self.num_indicators = len(os.listdir(self.fdir))
        self.num_vars = len(self.vars)

        self.start = kwargs.get('start', 'InputRequired')
        self.end = kwargs.get('end', 'InputRequired')

    def __read_data(self):
        data_list = []
        var_list = []
        for fname in os.listdir(self.fdir):
            fpath = os.path.sep.join((self.fdir, fname))

            _df = pd.read_excel(fpath, na_values='')
            for _, row in _df.iterrows():
                year = datetime.strftime(row['yearmonth'], '%Y')
                month = datetime.strftime(row['yearmonth'], '%m')
                yearmonth = '{}{}'.format(year, month)

                for var in row.keys():
                    if var == 'yearmonth':
                        continue
                    else:
                        data_list.append((yearmonth, var, row[var]))
                        var_list.append(var)

        return data_list, list(set(var_list))

    def __set_time_range(self, start, end):
        '''
        Attributes
        ----------
        start : str
            | YYYYMM
        end : str
            |YYYYMM
        '''

        start_dt = datetime.strptime(start, '%Y%m')
        end_dt = datetime.strptime(end, '%Y%m')
        return pd.date_range(start_dt, end_dt, freq='MS').strftime('%Y%m').tolist()

    def to_df(self, **kwargs):
        start = kwargs.get('start', self.start)
        end = kwargs.get('end', self.end)
        time_range = self.__set_time_range(start, end)

        _dict = defaultdict(list)
        _dict['yearmonth'] = time_range
        for yearmonth, var, value in sorted(self.data_list, key=lambda x:x[0], reverse=False):
            if yearmonth in time_range:
                try:
                    value2num = float(value)
                except ValueError:
                    value2num = 0
                _dict[var].append(value2num)
            else:
                continue

        _df = pd.DataFrame(_dict)
        return _df