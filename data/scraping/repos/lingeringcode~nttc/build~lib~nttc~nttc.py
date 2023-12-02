# -*- coding: utf-8 -*-
#!/usr/bin/python3

# NTTC (Name That Twitter Community!) A Tweets Topic Modeling Processor for Python 3
# by Chris Lindgren <chris.a.lindgren@gmail.com>
# Distributed under the BSD 3-clause license.
# See LICENSE.txt or http://opensource.org/licenses/BSD-3-Clause for details.

# WHAT IS IT?
# A set of functions that process and create topic models from a sample of community-detected Twitter networks' tweets.
# It assumes you seek an answer to the following questions:
#    1.) What communities persist or are ephemeral across periods in the copora, and when?
#    2.) What can these communities be named, based on their sources, targets, topics, and top-RT'd tweets?
#    3.) Of these communities, what are their topics over time?
# Accordingly, it assumes you have a desire to investigate tweets from each detected community across
# already defined periodic episodes with the goal of naming each community AND examining their
# respective topics over time in the corpus.

# It functions only with Python 3.x and is not backwards-compatible.

# Warning: nttc performs no custom error-handling, so make sure your inputs are formatted properly! If you have questions, please let me know via email.
import arrow
import ast
from collections import Counter
import csv
import emoji
import functools
import hdbscan
import itertools as it
import matplotlib.pyplot as plt
from MulticoreTSNE import MulticoreTSNE as TSNE
import networkx as nx
import numpy as np
import operator
import os
from os import listdir
from os.path import join, isfile
import pandas as pd
from pprint import pprint
import re
import seaborn as sns
import joblib
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.decomposition import TruncatedSVD
from stop_words import get_stop_words
import string
import sys
from tqdm import tqdm_notebook as tqdm
import tsm


# Stopwords
# Import stopwords with nltk.
import nltk
from nltk.corpus import stopwords
from nltk.util import everygrams
from nltk.tokenize.casual import TweetTokenizer

# Topic-Modeling imports
from pprint import pprint

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *

class allPeriodsObject:
    '''an object class with an attribute dict that stores per Period network data of nodes and edges in respective Dataframes'''
    def __init__(self, all_period_networks_dict=None, period_dates=None, info_hubs=None):
        self.all_period_networks_dict = all_period_networks_dict
        self.period_dates = period_dates
        self.info_hubs = info_hubs

class periodObject:
    '''an object class with attributes that store per Community subgraph properties'''
    def __init__(self, comm_nums=None, subgraphs_dict=None):
        self.comm_nums = comm_nums
        self.subgraphs_dict = subgraphs_dict

class communitiesObject:
    '''an object class with attributes for various matched-community data and metadata'''
    def __init__(self, content_slice=None, split_docs=None, id2word=None, texts=None,
                 corpus=None, readme=None, model=None, perplexity=None, coherence=None,
                 top_rts=None, top_mentions=None, full_hub=None):
        self.content_slice = content_slice
        self.split_docs = split_docs
        self.id2word = id2word
        self.texts = texts
        self.corpus = corpus
        self.readme = readme
        self.model = model
        self.perplexity = perplexity
        self.coherence = coherence
        self.top_rts = top_rts
        self.top_mentions = top_mentions
        self.full_hub = full_hub

class communityGroupsObject:
    '''an object class with attributes for various matched-community data and metadata'''
    def __init__(self, best_matches_mentions=None, best_matches_rters=None, sorted_filtered_mentions=None, sorted_filtered_rters=None, groups_mentions=None, groups_rters=None):
        self.best_matches_mentions = best_matches_mentions
        self.best_matches_rters = best_matches_rters
        self.sorted_filtered_mentions = sorted_filtered_mentions
        self.sorted_filtered_rters = sorted_filtered_rters
        self.groups_mentions = groups_mentions
        self.groups_rters = groups_rters

class clusteringObj:
    '''object class with attributes for conducting kmeans clustering'''
    def __init__(self, reduced_sample=None, tokens=None, top_grams=None, unique_obs_cnt=None, 
                 vector=None, u_map=None, u_users=None, u_texts=None, matrix=None, fit_file=None, 
                 matrix_2d=None, best_k=None, km_model=None, km_plottable=None):
        self.reduced_sample = reduced_sample
        self.tokens = tokens
        self.top_grams = top_grams
        self.unique_obs_cnt = unique_obs_cnt
        self.vector = vector
        self.u_map = u_map
        self.u_users = u_users
        self.u_texts = u_texts
        self.matrix = matrix
        self.fit_file = fit_file
        self.matrix_2d = matrix_2d
        self.best_k = best_k
        self.km_model = km_model
        self.km_plottable = km_plottable

##################################################################

## General Functions

##################################################################
'''
    Initialize initializeCluster
'''
def initializeCluster():
    return clusteringObj()

'''
    Initialize allPeriodsObject
'''
def initializeAPO():
    return allPeriodsObject()

'''
    Initialize periodObject
'''
def initializePO():
    return periodObject()

'''
    Initialize communitiesObject
'''
def initializeCGO():
    return communitiesObject()

'''
    Initialize communityGroupsObject
'''
def initializeCGO():
    return communityGroupsObject()

'''
    Load CSV data
'''
def get_csv(sys_path, __file_path__, dtype_dict):
    df_tw = pd.read_csv(join(sys_path, __file_path__),
                               delimiter=',',
                               dtype=dtype_dict,
                               error_bad_lines=False)
    return df_tw

'''
    batch_csv(): Merge a folder of CSV files into either one allPeriodsObject 
        that stores a dict of all network nodes and edges per Period, or returns 
        only the aforementioned dict, if no object is passed as an arg.
    Args:
        - path= String of path to directory with files
        - all_po= Optional instantiated allPeriodsObject
    Return: Either
        - Dict of per Period nodes and edges, or
        - allPeriodsObject with Dict stored as property
'''
def batch_csv(**kwargs):
    # Pattern for period number from filename
    re_period = r"(\d{1,2})"
    periods = []
    network_dicts = {}
    
    # Write list of periods
    for f in listdir(kwargs['path']):
        period_num = re.search(re_period, f)
        if period_num:
            if not periods:
                periods.append(period_num.group(0))
            elif period_num.group(0) not in periods:
                periods.append(period_num.group(0))
    
    # Listify files within path and ignore hidden files
    list_of_files = [f for f in listdir(kwargs['path']) if not f.startswith('.') and isfile(join(kwargs['path'], f))]
    
    # If period column exists, consolidate
    if 'p_col_exists' in kwargs:
        df_obj = pd.concat([pd.read_csv(file, index=False) for file in list_of_files])

        print(
        'Batch DF merge complete. First 5 rows:\n\n',
        df_obj[:5],
        '\n\nDF summary:\n\n',
        df_obj.describe()
        )
    # If period column doesn't exist, make it from filenames
    else:
        re_node = r"(node)"
        re_edge = r"(edge)"
        # Consolidate all CSV files into one Dataframe
        for a_file in list_of_files:
            new_df = pd.read_csv(join(kwargs['path'], a_file))
            node_check = re.search(re_node, a_file)
            edge_check = re.search(re_edge, a_file)
            period_num = re.search(re_period, a_file)
            if node_check:
                if len(network_dicts) == 0:
                    network_dicts.update({period_num.group(0): {'nodes': new_df}})
                elif period_num.group(0) not in network_dicts:
                    network_dicts.update({period_num.group(0): {'nodes': new_df}})
                elif period_num.group(0) in network_dicts:
                    network_dicts[period_num.group(0)].update({'nodes': new_df})
            elif edge_check:
                if len(network_dicts) == 0:
                    network_dicts.update({period_num.group(0): {'edges': new_df}})
                elif period_num.group(0) not in network_dicts:
                    network_dicts.update({period_num.group(0): {'edges': new_df}})
                elif period_num.group(0) in network_dicts:
                    network_dicts[period_num.group(0)].update({'edges': new_df})
        if 'all_po' in kwargs:
            kwargs['all_po'].all_period_networks_dict = network_dicts
            return kwargs['all_po']
        if 'all_po' not in kwargs:
            return network_dicts

'''
    write_csv(): Writes Dataframe input as an output CSV file.
'''
def write_csv(dal, sys_path, __file_path__):
    dal.to_csv(join(sys_path, __file_path__),
                                sep=',',
                                encoding='utf-8',
                                index=False)
    print(__file_path__, ' written to ', sys_path)

##################################################################

## KMeans Clustering Functions
##  - Some functions below are modified from the MIT-licensed 
##      Twitter Dev resource:
## https://twitterdev.github.io/do_more_with_twitter_data/clustering-users.html

##################################################################

def sample_reducer(sample_type, dict_samples, columns_list):
    if sample_type == 'single':
        for m in dict_samples:
            if len(dict_samples[m]['sample']) > 0:
                tweet_df = dict_samples[m]['sample'][columns_list]
                clusterObj = initializeCluster()
                clusterObj.reduced_sample = tweet_df
                dict_samples[m]['obj'] = clusterObj
    elif sample_type == 'multiple':
        for p in dict_samples:
            for m in dict_samples[p]:
                if len(dict_samples[p][m]['sample']) > 0:
                    tweet_df = dict_samples[p][m]['sample'][columns_list]
                    clusterObj = initializeCluster()
                    clusterObj.reduced_sample = tweet_df
                    dict_samples[p][m]['obj'] = clusterObj
    
    return dict_samples

'''
    replace_urls: Replace URLs in strings. See also: ``bit.ly/PyURLre``
        Args:
            in_string (str): string to filter
            replacement (str or None): replacment text. defaults to '<-URL->'

        Returns:
            str
'''
def replace_urls(in_string, replacement=None):
    replacement = '<-URL->' if replacement is None else replacement
    pattern = re.compile('(https?://)?(\w*[.]\w+)+([/?=&]+\w+)*')
    return re.sub(pattern, replacement, in_string)

def df_datetime_converter(sample_type, dict_samples, col_name):
    if sample_type == 'single':
        for m in dict_samples:
            if 'obj' in dict_samples[m]:
                dict_samples[m]['obj'].reduced_sample.loc[:, col_name] = pd.to_datetime(
                    dict_samples[m]['obj'].reduced_sample[col_name]
                )
    elif sample_type == 'multiple':
        for p in dict_samples:
            for m in dict_samples[p]:
                if 'obj' in dict_samples[p][m]:
                    dict_samples[p][m]['obj'].reduced_sample.loc[:, col_name] = pd.to_datetime(
                        dict_samples[p][m]['obj'].reduced_sample[col_name]
                    )
    return dict_samples

def plot_tweet_counts(sample_type, dict_samples, zoom, col_list):
    if sample_type == 'single':
        for m in dict_samples:
            if 'obj' in dict_samples[m]:
                title = 'Module ' + m + zoom + ' Counts'
                (dict_samples[m]['obj'].reduced_sample[col_list]
                    .set_index('date')
                    .resample(zoom)
                    .count()
                    .rename(columns=dict(tweet=title))
                    .plot()
                )
    elif sample_type == 'multiple':
        for p in dict_samples:
            for m in dict_samples[p]:
                if 'obj' in dict_samples[p][m]:
                    title = 'Period ' + p + ',Module ' + m + ': ' + zoom + ' Counts'
                    (dict_samples[p][m]['obj'].reduced_sample[col_list]
                        .set_index('date')
                        .resample(zoom)
                        .count()
                        .rename(columns=dict(tweet=title))
                        .plot()
                    )

def get_all_tokens(tweet_list):
    # concat corpus
    all_text = ' '.join((t for t in tweet_list))
    # tokenize
    tokens = (TweetTokenizer(preserve_case=False,
                            reduce_len=True,
                            strip_handles=False)
              .tokenize(all_text))
    # remove symbol-only tokens
    tokens = [tok for tok in tokens if not tok in string.punctuation]
    return tokens

def tokenize_em(sample_type, dict_samples, col):
    if sample_type == 'single':
        for m in dict_samples:
            if 'obj' in dict_samples[m]:
                listicle = dict_samples[m]['obj'].reduced_sample[col].values.tolist()
                clean_listicle = []
                for l in listicle:
                    cl = replace_urls(l)
                    clean_listicle.append(cl)
                tokens = get_all_tokens(clean_listicle)
                print('Total number of tokens: {}'.format(len(tokens)))
                dict_samples[m]['obj'].tokens = tokens
    elif sample_type == 'multiple':
        for p in dict_samples:
            for m in dict_samples[p]:
                if 'obj' in dict_samples[p][m]:
                    listicle = dict_samples[p][m]['obj'].reduced_sample[col].values.tolist()
                    clean_listicle = []
                    for l in listicle:
                        cl = replace_urls(l)
                        clean_listicle.append(cl)
                    tokens = get_all_tokens(clean_listicle)
                    print('Total number of tokens: {}'.format(len(tokens)))
                    dict_samples[p][m]['obj'].tokens = tokens
    
    return dict_samples

def top_gram_counter(sample_type, dict_samples, min_len, max_len):
    if sample_type == 'single':
        for m in dict_samples:
            if 'obj' in dict_samples[m]:
                top_grams = Counter(everygrams(dict_samples[m]['obj'].tokens, min_len=min_len, max_len=max_len))
                dict_samples[m]['obj'].top_grams = top_grams
    elif sample_type == 'multiple':
        for p in dict_samples:
            for m in dict_samples[p]:
                if 'obj' in dict_samples[p][m]:
                    top_grams = Counter(everygrams(dict_samples[p][m]['obj'].tokens, min_len=min_len, max_len=max_len))
                    dict_samples[p][m]['obj'].top_grams = top_grams
    
    return dict_samples

def unique_obs_counter(sample_type, dict_samples):
    if sample_type == 'single':
        for m in dict_samples:
            if 'obj' in dict_samples[m]:
                unique_obs_cnt = len(set(dict_samples[m]['obj'].reduced_sample['username']))
                dict_samples[m]['obj'].unique_obs_cnt = unique_obs_cnt
    elif sample_type == 'multiple':
        for p in dict_samples:
            for m in dict_samples[p]:
                if 'obj' in dict_samples[p][m]:
                    unique_obs_cnt = len(set(dict_samples[p][m]['obj'].reduced_sample['username']))
                    dict_samples[p][m]['obj'].unique_obs_cnt = unique_obs_cnt
    
    return dict_samples

'''
    Generates punctuation 'words' up to ``max_length`` characters.
'''
def make_punc_stopwords(max_length=4):
    def punct_maker(length):
        return ((''.join(x) for x in it.product(string.punctuation,
                                                repeat=length)))
    words = it.chain.from_iterable((punct_maker(length)
                                    for length in range(max_length+1)))
    return list(words)

'''
    Convert `in_string` of text to a list of tokens using NLTK's TweetTokenizer
'''
def my_tokenizer(in_string):
    # reasonable, but adjustable tokenizer settings
    tokenizer = TweetTokenizer(preserve_case=False,
                               reduce_len=True,
                               strip_handles=False)
    tokens = tokenizer.tokenize(in_string)
    return tokens

def tm_vectorizer(sample_type, dict_samples, stop_words):
    if sample_type == 'single':
        for m in dict_samples:
            if 'obj' in dict_samples[m]:
                try:
                    vec = TfidfVectorizer(preprocessor=replace_urls,
                            tokenizer=my_tokenizer,
                            stop_words=stop_words,
                            max_features=dict_samples[m]['obj'].unique_obs_cnt/100,
                            )
                    dict_samples[m]['obj'].vector = vec
                except ValueError as e:
                    print('\nFor module', m, 'assign as None. ','ValueError ', e)
                    dict_samples[m]['obj'].vector = None
    elif sample_type == 'multiple':
        for p in dict_samples:
            for m in dict_samples[p]:
                if 'obj' in dict_samples[p][m]:
                    try:
                        vec = TfidfVectorizer(preprocessor=replace_urls,
                            tokenizer=my_tokenizer,
                            stop_words=stop_words,
                            max_features=dict_samples[p][m]['obj'].unique_obs_cnt/100,
                            )
                        dict_samples[p][m]['obj'].vector = vec
                    except ValueError as e:
                        print('\nFor Period', p, 'Module', m, 'assign as None.', 'ValueError ', e)
                        dict_samples[p][m]['obj'].vector = None
    
    return dict_samples

'''
    unique_observ_mapper: Write Dict of unique observations for kmeans analysis:
        - Structure: {'observation_name': 'variable_name'}
'''
def unique_observ_mapper(sample_type, dict_samples, observation, variable):
    if sample_type == 'single':
        for m in dict_samples:
            if 'obj' in dict_samples[m]:
                # WRITE UNIQUE MAP
                unique_map = {}
                for row in dict_samples[m]['obj'].reduced_sample.to_dict('records'):
                    unique_map.update({
                        str(row[observation]): row[variable]}
                    )
                # WRITE UNIQUE USER AND TEXT LISTS
                unique_users = []
                unique_texts = []
                for user,text in unique_map.items():
                    unique_users.append(user)
                    if text is None:
                        # special case for empty text
                        text = ''
                    unique_texts.append(text)

                dict_samples[m]['obj'].u_map = unique_map
                dict_samples[m]['obj'].u_users = unique_users
                dict_samples[m]['obj'].u_texts = unique_texts
                print(
                    m,
                    len(dict_samples[m]['obj'].u_map),
                    len(dict_samples[m]['obj'].u_users),
                    len(dict_samples[m]['obj'].u_texts)
                )
    elif sample_type == 'multiple':
        for p in dict_samples:
            for m in dict_samples:
                if 'obj' in dict_samples[p][m]:
                    # WRITE UNIQUE MAP
                    unique_map = {}
                    for row in dict_samples[p][m]['obj'].reduced_sample.to_dict('records'):
                        unique_map.update({
                            str(row[observation]): row[variable]}
                        )
                    # WRITE UNIQUE USER AND TEXT LISTS
                    unique_users = []
                    unique_texts = []
                    for user,text in unique_map.items():
                        unique_users.append(user)
                        if text is None:
                            # special case for empty text
                            text = ''
                        unique_texts.append(text)
                    print(
                        p,m,
                        len(unique_map),
                        len(unique_users),
                        len(unique_texts)
                    )
                    dict_samples[p][m]['obj'].u_map = unique_map
                    dict_samples[p][m]['obj'].u_users = unique_users
                    dict_samples[p][m]['obj'].u_texts = unique_texts
    
    return dict_samples

def calc_matrix(sample_type, dict_samples):
    if sample_type == 'single':
        for m in dict_samples:
            if ('obj' in dict_samples[m]) and (dict_samples[m]['obj'].vector is not None):
                    dict_samples[m]['obj'].matrix = dict_samples[m]['obj'].vector.fit_transform(
                        dict_samples[m]['obj'].u_texts
                    )
    elif sample_type == 'multiple':
        for p in dict_samples:
            for m in dict_samples[p]:
                if ('obj' in dict_samples[p][m]) and (dict_samples[p][m]['obj'].vector is not None):
                        dict_samples[p][m]['obj'].matrix = dict_samples[p][m]['obj'].vector.fit_transform(
                            dict_samples[p][m]['obj'].u_texts
                        )
    return dict_samples

'''
    kmeans_plotter:
    Args:
    Returns: 
'''
def kmeans_plotter(matrix, title, seed, kv_list):
    # Sort k values just in case
    kv_list = sorted(kv_list, key=int)

    # track a couple of metrics
    sil_scores = []
    inertias = []

    # fit the models, save the evaluation metrics from each run
    for k in tqdm(kv_list):
        # Append k values to title
        title = title + ' ' + str(k)
        print('fitting model for {} clusters'.format(k))
        model = KMeans(n_clusters=k, n_jobs=-1, random_state=seed)
        model.fit(matrix)
        labels = model.labels_
        sil_scores.append(silhouette_score(matrix, labels))
        inertias.append(model.inertia_)

    # plot the quality metrics for inspection    
    fig, ax = plt.subplots(2, 1, sharex=True)

    plt.subplot(211)
    plt.plot(kv_list, inertias, 'o--')
    plt.ylabel('inertia')
    plt.title(title)

    plt.subplot(212)
    plt.plot(kv_list, sil_scores, 'o--')
    plt.ylabel('silhouette score')
    plt.xlabel('k')
    plt.show()

def compare_kmeans(sample_type, dict_samples, seed, kv_list):
    if sample_type == 'single':
        for m in dict_samples:
            if ('obj' in dict_samples[m]) and (dict_samples[m]['obj'].vector is not None):
                title = 'Module '+m+' Kmeans parameter search'
                kmeans_plotter(dict_samples[m]['obj'].matrix, title, seed, kv_list)
    elif sample_type == 'multiple':
        for p in dict_samples:
            for m in dict_samples[p]:
                if ('obj' in dict_samples[p][m]) and (dict_samples[p][m]['obj'].vector is not None):
                    title = 'Period '+p+' Module '+m+' Kmeans parameter search'
                    kmeans_plotter(dict_samples[p][m]['obj'].matrix, title, seed, kv_list)

def fit_filename_writer(sample_type, dict_samples, file_primer):
    file = ''
    if sample_type == 'single':
        for m in dict_samples:
            if ('obj' in dict_samples[m]) and (dict_samples[m]['obj'].vector is not None):
                file = file_primer + m + '_2d.npy'
                dict_samples[m]['obj'].fit_file = file
    elif sample_type == 'multiple':
        for p in dict_samples:
            for m in dict_samples[p]:
                if ('obj' in dict_samples[p][m]) and (dict_samples[p][m]['obj'].vector is not None):
                    file = file_primer + p+m + '_2d.npy'
                    dict_samples[p][m]['obj'].fit_file = file
    return dict_samples

def trial_fit_tsne(sample_type, dict_samples, seed):
    if sample_type == 'single':
        for m in tqdm(dict_samples):
            if ('obj' in dict_samples[m]) and (dict_samples[m]['obj'].vector is not None):
                try:
                    matrix_2d = np.load(dict_samples[m]['obj'].fit_file)
                    logging.warning("loading cached TSNE file")
                    dict_samples[m]['obj'].matrix_2d = matrix_2d
                except FileNotFoundError:
                    logging.warning("Fitting TSNE")
                    tsne = TSNE(n_components=2,
                                n_jobs=-1,
                                random_state=seed)
                    matrix_2d = tsne.fit_transform(dict_samples[m]['obj'].matrix.todense())
                    dict_samples[m]['obj'].matrix_2d = matrix_2d
                    np.save(dict_samples[m]['obj'].fit_file, matrix_2d)
    elif sample_type == 'multiple':
        for p in tqdm(dict_samples):
            for m in dict_samples[p]:
                if ('obj' in dict_samples[p][m]) and (dict_samples[p][m]['obj'].vector is not None):
                    try:
                        matrix_2d = np.load(dict_samples[p][m]['obj'].fit_file)
                        logging.warning("loading cached TSNE file")
                        dict_samples[p][m]['obj'].matrix_2d = matrix_2d
                    except FileNotFoundError:
                        logging.warning("Fitting TSNE")
                        tsne = TSNE(n_components=2,
                                    n_jobs=-1,
                                    random_state=seed)
                        matrix_2d = tsne.fit_transform(dict_samples[p][m]['obj'].matrix.todense())
                        dict_samples[p][m]['obj'].matrix_2d = matrix_2d
                        np.save(dict_samples[p][m]['obj'].fit_file, matrix_2d)
    return dict_samples

'''
    get_plottable_df: Combine the necessary variables for the 2d tsne chart.
'''
def get_plottable_df(labels, users, texts, two_d_coords):
    # set up color palette
    num_labels = len(set(labels))
    colors = sns.color_palette('hls', num_labels).as_hex()
    color_lookup = {v:k for k,v in zip(colors, set(labels))}
    # combine data into a single df
    df = pd.DataFrame({'username': users,
                       'text': texts,
                       'label': labels,
                       'x_val': two_d_coords[:,0],
                       'y_val': two_d_coords[:,1],
                      })
    # convert labels to colors
    df['color'] = list(map(lambda x: color_lookup[x], labels))
    return df

def get_plottable_controller(sample_type, dict_samples):
    if sample_type == 'single':
        for m in dict_samples:
            if ('obj' in dict_samples[m]) and (dict_samples[m]['obj'].vector is not None):
                # set up color palette
                num_labels = len(set(dict_samples[m]['obj'].km_model.labels_))
                colors = sns.color_palette('hls', num_labels).as_hex()
                color_lookup = {v:k for k,v in zip(colors, set(dict_samples[m]['obj'].km_model.labels_))}
                # combine data into a single df
                df = pd.DataFrame({'username': dict_samples[m]['obj'].u_users,
                                   'text': dict_samples[m]['obj'].u_texts,
                                   'label': dict_samples[m]['obj'].km_model.labels_,
                                   'x_val': dict_samples[m]['obj'].matrix_2d[:,0],
                                   'y_val': dict_samples[m]['obj'].matrix_2d[:,1],
                                  })
                # convert labels to colors
                df['color'] = list(map(lambda x: color_lookup[x], dict_samples[m]['obj'].km_model.labels_))
                dict_samples[m]['obj'].km_plottable = df
    elif sample_type == 'multiple':
        for p in dict_samples:
            for m in dict_samples[p]:
                if ('obj' in dict_samples[p][m]) and (dict_samples[p][m]['obj'].vector is not None):
                    # set up color palette
                    num_labels = len(set(dict_samples[p][m]['obj'].km_model.labels_))
                    colors = sns.color_palette('hls', num_labels).as_hex()
                    color_lookup = {v:k for k,v in zip(colors, set(dict_samples[p][m]['obj'].km_model.labels_))}
                    # combine data into a single df
                    df = pd.DataFrame({'username': dict_samples[p][m]['obj'].u_users,
                                       'text': dict_samples[p][m]['obj'].u_texts,
                                       'label': dict_samples[p][m]['obj'].km_model.labels_,
                                       'x_val': dict_samples[p][m]['obj'].matrix_2d[:,0],
                                       'y_val': dict_samples[p][m]['obj'].matrix_2d[:,1],
                                      })
                    # convert labels to colors
                    df['color'] = list(map(lambda x: color_lookup[x], dict_samples[p][m]['obj'].km_model.labels_))
                    dict_samples[p][m]['obj'].km_plottable = df
    return dict_samples

'''
    cluster_sample: Helper function to display original texts for
    users modeled in cluster `idx`.
'''
def cluster_sample(orig_text, model, idx, preview=15):
    for i,idx in enumerate(np.where(model.labels_ == idx)[0]):
        print(orig_text[idx].replace('\n',' '))
        print()
        if i > preview:
            print('( >>> Truncated preview of cluster sample <<< )')
            break

##################################################################

## Infomap Data-Processing Functions

##################################################################
'''
    vert_lookup: Helper function for write_net_dict. It finds the
        matching username and returns the period_based ID.
'''
def vert_lookup(a, u_list):
    for u in u_list:
        if a == u[2]:
            return u[0]    

'''
    write_net_dict: Writes s Dict of vertices (nodes) and
        arcs (edges) in the Pajek file format (.net).
'''
def write_net_dict(**kwargs):
    index=1
    verts = []
    arcs = []
    for v in kwargs['vertices']:
        user = '\"'+str(v[1])+'\"'
        if len(v) == 3:
            verts.append([index, user, v[3]])
        else:
            verts.append([index, user, v[1]])
        index = index + 1

    for e in kwargs['keyed_edges']:
        if len(e) == 3:
            print(e[0], e[1], e[3])
        else:
            source = vert_lookup(e[0], verts)
            target = vert_lookup(e[1], verts)
            arcs.append([source,target])
    p_dict = {
        'vertices': verts,
        'arcs': arcs
    }
    return p_dict

'''
    write_net_txt(): Outputs .net file Pajek format:
        node_id nodes [optional weight]
        1 "user1"
        2 "user2"
        ...
        source target [optional weight]
        1 2
        2 1
        ...
'''
def write_net_txt(**kwargs):
    with open(join(kwargs['net_path'], kwargs['net_output']), "a") as f:
        print( '*Vertices', len(kwargs['vertices']), file=f )
        for v in kwargs['vertices']:
            user = '\"'+str(v[1])+'\"'
            if len(v) == 4:
                print(v[0], user, v[3], file=f)
            else:
                print(v[0], user, file=f)
        
        print( '*Arcs', len(kwargs['keyed_edges']), file=f )
        for e in kwargs['keyed_edges']:
            if e != None:
                if len(e) == 3:
                    print(e[0], e[1], e[3], file=f)
                else:
                    print(e[0], e[1], file=f)
    print('File', kwargs['net_output'], 'written to', kwargs['net_path'])

'''
    index_unique_users(): Take list of unique users and append IDs
'''
def index_unique_users(ul):
    index = 1
    new_un_list = []
    for un in ul:
        new_un_list.append( (index, un) )
        index = index + 1
    return new_un_list

'''
  check_protected_dataype_names: Verify that edge names don't conflict with Python protected datatypes.
      If they do, append 2 underscores to its end and log it.
'''
def check_protected_dataype_names(le):
    index = 0
    for e in le:
        if (str(e[0]) == 'nan' or str(e[0]) == 'None' or str(e[0]) == 'undefined' or str(e[0]) == 'null'):
            print('Source with encoded name found: ', e[0], ' at index ', index)
            e[0] = str(e[0])+'__'
            print('Renamed edge as ', e)
        elif (str(e[1]) == 'nan' or str(e[1]) == 'None' or str(e[1]) == 'undefined' or str(e[0]) == 'null'):
            print('Target with encoded name found: ', e[1], ' at index ', index)
            e[1] = str(e[1])+'__'
            print('Renamed edge as ', e)
        index=index+1
    return le

'''
    listify_unique_users(): Take edge list and create a list of unique users
'''
def listify_unique_users(**kwargs):
    user_list = []
    status_counter = 0
    index = 0
    print('Starting the listifying process. This may take some time, depending upon the file size.')
    for source, target in kwargs['edges']:
        if status_counter == 50000:
            print(index, 'entries processed.')
            status_counter = 0
        if (source not in user_list and target not in user_list):
            user_list.append(source)
            user_list.append(target)
        elif (source in user_list and target not in user_list):
            user_list.append(target)
        elif (source not in user_list and target in user_list):
            user_list.append(source)
        index = index + 1
        status_counter = status_counter + 1
    print('Listifying complete!')
    return user_list

'''
    target_part_lookup(): Lookup target in unique list and return
    to netify_edges()
'''
def target_part_lookup(nu_list, target):
    for n in nu_list:
        if n[1] == target:
            return n[0] #Return target

'''
    netify_edges(); Accepts list of lists (edges) and replaces the
        usernames with their unique IDs. This prepares output for the
        infomap code system.

    Args:
        - list_edges
        - unique_list
        - 
'''
def netify_edges(**kwargs):
    position = 0
    status_counter = 0
    new_edges_list = []
    print('Started revising edges to net format. This may take some time, depending upon the file size.')
    for source in kwargs['list_edges']:
        new_edge = None
        for un in kwargs['unique_list']:
            if status_counter == 50000:
                print(position, 'entries processed.')
                status_counter = 0
            if source[0] == un[1]:
                src = un[0]
                tar = target_part_lookup(
                    kwargs['unique_list'],
                    kwargs['list_edges'][position][1]) #send target for retrieval
                new_edge = [src, tar]
        new_edges_list.append(new_edge)
        position = position + 1
        status_counter = status_counter + 1
    print('Finished revising the edge list.')
    return new_edges_list

'''
    read_map_or_ftree: Helper function for infomap_hub_maker. Slices period's .map
    into their line-by-line indices and returns a dict of those values for use.
'''
def read_map_or_ftree(**kwargs):
    lines = [line.rstrip('\n') for line in open(join(kwargs['path'], kwargs['file']))]
    return lines

'''
    infomap_edge_data_lister: Listifies edge dataframe to include period and community
        module information. Easier to transform as new DF and export as CSV.
        Args:
            - dict_map: Saturated Dict from ftree_edge_maker().
            - p_sample: Integer. Total periods to sample.
            - m_sample: Integer. Total modules to sample.
        Return:
            - list_edge_data: List of lists of edge data.
'''
def infomap_edge_data_lister(dict_map,p_sample,m_sample):
    list_edge_data = []
    for p in range(1, p_sample):
        for m in range(1, m_sample):
            pm_df = dict_map[str(p)]['indices']['ftree_links'][str(m)]['df_edges']
            # Parse DF for export
            for d in pm_df.values.tolist():
                list_edge_data.append([str(p), str(m), d[0][:-1], d[1][:-1], d[2]])
    return list_edge_data

'''
    ftree_edge_maker: Processes each period's module edge data and stores as a DataFrame.
        Args: 
            - dict_map: Dict. Saturated dict from batchmap and indices_getter functions.
        Return:
            - dict_map: Dict. dict_map now includes edge data per Period per Module as
                pandas DataFrames.
'''
def ftree_edge_maker(dict_map):
    re_edge_data = r"(\d{1,10}\s)(\d{1,}\s)(\d\.\d{1,}e-\d{1,}|\d{1,}\.\d{1,})"
    
    re_ignore_ftree_links_mods = r"\*Links\s\d{1,}(\:\d{1,}){1,}\s" #ignore these hierarchical lines

    # Go through each period
    for p in dict_map:
        print('Processing edge data for period', p)
        # Each module's links
        for m in dict_map[p]['indices']['ftree_links']:
            start = 0
            end = 0
            
            # If length of indices == 1 or 0, then there are no edges
            if len(dict_map[p]['indices']['ftree_links'][m]['indices']) == 0 or len(dict_map[p]['indices']['ftree_links'][m]['indices']) == 1:
                break
                
            elif dict_map[p]['indices']['ftree_links'][m]['indices'][0] < dict_map[p]['indices']['ftree_links'][m]['indices'][1]:
                start = dict_map[p]['indices']['ftree_links'][m]['indices'][0]
                end = dict_map[p]['indices']['ftree_links'][m]['indices'][1]
            
            # If start equal to end, then there are no edges.
            elif dict_map[p]['indices']['ftree_links'][m]['indices'][0] == dict_map[p]['indices']['ftree_links'][m]['indices'][1]:
                break
            
            list_edges = []
            for l in dict_map[p]['lines'][start:end]:
                
                # Check for ftree submod links and ignoore those lines,
                # since they're not edges, but metadata
                re_check_sub_ftree = re.findall(re_ignore_ftree_links_mods, l)
                if len(re_check_sub_ftree) == 0:
                
                    group_matches = re.findall(re_edge_data, l)

                    list_edges.append([group_matches[0][0][:-1],group_matches[0][1][:-1],group_matches[0][2]])
            
            df = pd.DataFrame(list_edges,columns=['source','target','directed_count'])
            
            if len(df) > 0:
                # Append network edge data as dataframe
                dict_map[p]['indices']['ftree_links'][m]['df_edges'] = df
            
            else:
                # Demarcate no edge data as Integer 0
                dict_map[p]['indices']['ftree_links'][m]['df_edges'] = 0

    print('Processing complete!')
    return dict_map

'''
'''
def get_link_indices_info(root_check, index, l):
    re_ftree_link_sum_stats = r"(\s\d\.\d{1,}|\d\.\d{1,}e-\d{1,}|\s\d)(\s\d{1,})(\s\d{1,})"
    dict_entry = {}

    if root_check == True:
        link_root_begin_index = index+1
        link_stat_match = re.findall(re_ftree_link_sum_stats, l)
        exit_flow = link_stat_match[0][0][1:]
        num_edges = link_stat_match[0][1][1:]
        num_children = link_stat_match[0][2][1:]
        key = l[6:11]
        dict_entry.update({
            '1': {
                'exit_flow': exit_flow,
                'num_edges': num_edges,
                'num_children': num_children,
                'indices': [link_root_begin_index]
            }
        })
        return dict_entry
    if root_check == False:
        re_ftree_links_mod_num = r"\*Links\s\d{1,10}\s"
        # Indices
        link_mod_begin_index = index+1
        # Match up desired data
        mod_header_match = re.findall(re_ftree_links_mod_num, l)
        link_stat_match = re.findall(re_ftree_link_sum_stats, l)
        exit_flow = link_stat_match[0][0][1:]
        num_edges = link_stat_match[0][1][1:]
        num_children = link_stat_match[0][2][1:]
        # Assign module number as dict key
        key = mod_header_match[0][6:-1]
        key = int(key)+1
        key = str(key)
        # Update current link info
        dict_entry.update({
            key: {
                'exit_flow': exit_flow,
                'num_edges': num_edges,
                'num_children': num_children,
                'indices': [link_mod_begin_index]
            }
        })
        return dict_entry

'''
    indices_getter: Helper function for batch_map. Parses each line in the file
    and returns a list of lists, where each sublists is a line in the file.
        Args:
            - file_type: String. 'ftree' or 'map' to denote input file format.
            - lines: List. Listified file, where each row is a String to be parsed.
        Return:
            - dict_indices: Dict. Contains indices and basic stat information
'''
def indices_getter(file_type, lines):
    re_modules = r"\*Modules\s\d{1,}"
    re_nodes = r"\*Nodes\s\d{1,}"
    re_links = r"\*Links\s\d{1,}"
    re_ftree = r"\#\spath\sflow\sname\snode\:" #Ftree start to modules
    re_ftree_links_hd_1 = r"\*Links\sdirected" #End of mods, start of first Links
    re_ftree_links_mods_root = r"\*Links\sroot\s\d{1,}\.\d{1,}\s\d{1,}\s\d{1,}" #root link
    re_ftree_links_mods = r"\*Links\s\d{1,}\s\d{1,}\.\d{1,}\s\d{1,}\s\d{1,}|\*Links\s\d{1,}\s\d{1,}\.\d{1,}e-\d{1,}\s\d{1,}\s\d{1,}|\*Links\s\d{1,}\s\d{1,}\s\d{1,}\s\d{1,}|\*Links\s\d{1,}\s-\d{1,}\.\d{1,}e-\d{1,}\s\d{1,}\s\d{1,}" #all other links
    
    index = 0
    mod_index = 0
    node_index = 0
    link_index = 0
    mod_list = []
    node_list = []
    link_list = []
    dict_indices = {}
    
    if file_type == 'map':
        for l in lines:
            mod_match = re.match(re_modules, l)
            node_match = re.match(re_nodes, l)
            link_match = re.match(re_links, l)
            
            if mod_match != None:
                mod_index = index
            elif node_match != None:
                node_index = index
            elif link_match != None:
                link_index = index
                link_list.append([link_index+1, len(lines)-1])
            index = index + 1
        
        mod_list.append([mod_index+1, node_index-1])
        node_list.append([node_index+1, link_index-1])
        
        dict_indices = {
            'modules': mod_list[0],
            'nodes': node_list[0],
            'links': link_list[0]
        }
    if file_type == 'ftree':
        # PARSE ALL MODULES & LINKS
        ftree_mods_begin_index = 0
        links_begin_index = 0
        dict_ftree_mod_edges = {}

        for l in lines:
            ftree_match = re.match(re_ftree, l)
            link_match = re.match(re_ftree_links_hd_1, l)
            link_root_mod_match = re.match(re_ftree_links_mods_root, l)
            link_num_mod_match = re.match(re_ftree_links_mods, l)

            #If match, assign current index for indices
            if ftree_match != None:
                ftree_mods_begin_index = index
            elif link_match != None:
                links_begin_index = index
            elif link_root_mod_match != None:
                rc = True
                dict_info = get_link_indices_info(rc, index,l)
                dict_ftree_mod_edges.update(dict_info)

            elif link_num_mod_match != None:
                rc = False
                dict_info = get_link_indices_info(rc, index, l)
                dict_ftree_mod_edges.update(dict_info)
                
                # Update previous link indices
                re_ftree_links_mod_num = r"\*Links\s\d{1,10}\s"
                mod_header_match = re.findall(re_ftree_links_mod_num, l)
                key = mod_header_match[0][6:-1]
                key = int(key)+1
                key = str(key)
                last_key = int(key)-1
                last_key = str(last_key)
                
                # Assign index from last key
                try:
                    dict_ftree_mod_edges[last_key]['indices'].append(index)
                except KeyError as e:
                    print('KeyError ',e)

            index = index + 1
        
        # Assign indices to list
        dict_indices = {
            'ftree_modules': [ftree_mods_begin_index+1, links_begin_index-1],
            'ftree_links': dict_ftree_mod_edges
        }

    return dict_indices

'''
    batch_map: Retrieves all map files in a single directory. It assumes
    that you have only the desired files in said directory. Returns a dict
    of each files based on their naming scheme with custom regex pattern.
    Each key denotes the file and its values are list of lists, where each sublist
    is a lines in the file.
    - regex= Regular expression for filename scheme
    - path= String. Path for directory with .map or .ftree files
    - file_type= String. File format type, such as 'map' or 'ftree'
    - data_type= String. Type of data to parse, such as 'modules' or 'links'
'''
def batch_map(**kwargs):
    # Pattern for period number from filename
    re_period = kwargs['regex']
    periods = []
    map_dicts = {}
    indices = {}
    # Listify files within path, ignore hidden files, look for only defined type
    list_of_files = [f for f in listdir(kwargs['path']) if not f.startswith('.') and f.endswith(kwargs['file_type']) and isfile(join(kwargs['path'], f))]
    
    # Write list of periods
    for f in list_of_files:
        period_num = re.findall(re_period, f)
        if period_num[0]:
            if not periods:
                periods.append(period_num[0])
            elif period_num[0] not in periods:
                periods.append(period_num[0])
    
    # Write per Period dict with each file as List of lines to parse
    map_dicts = {}
    sorted_list_of_files = sorted(list_of_files)
    for f in sorted_list_of_files:
        period_num = re.search(re_period, f)
        p = period_num[0]
        lines = read_map_or_ftree(path=kwargs['path'], file=f)

        if kwargs['file_type'] == 'map':
            indices = indices_getter(kwargs['file_type'], lines)
            map_dicts.update({ p: 
                        {
                            'lines': lines,
                            'indices': indices
                        } 
            })
        elif kwargs['file_type'] == 'ftree':
            indices = indices_getter(kwargs['file_type'], lines)
            map_dicts.update({ p: 
                        {
                            'lines': lines,
                            'indices': indices
                        } 
            })
    
    return map_dicts

'''
    network_organizer: Organizes infomap .ftree network edge and node data
        into Dict.
        Args:
            - m_edges: DataFrame. Per period module edge data
            - m_mod: List of Dicts. Per period list of module data 
        Return:
            - return_dict: Dict. Network node and edge data with names:
                { 
                    return_dict: 
                        '1':{
                            '1':{
                                'nodes': DataFrame,
                                'edges': Dataframe
                            }, ...
                        }, ...
                    }
                }
'''
def network_organizer(m_edges, m_mod):
    new_list_nodes = []
    new_dict_edges = []
    return_dict = {}
    eds = m_edges.to_dict('records')
    for e in tqdm(eds):
        for n in m_mod:
            if str(e['source']) == str(n['node']):
                e['source_name'] = n['name']
                if n['name'] not in new_list_nodes:
                    new_list_nodes.append(n['name'])
                for nn in m_mod:
                    if str(e['target']) == str(nn['node']):
                        e['target_name'] = nn['name']
                        new_dict_edges.append(e)
                        if nn['name'] not in new_list_nodes:
                            new_list_nodes.append(n['name'])
                        break
    
    df_full_nodes = pd.DataFrame(new_list_nodes, columns=['username'])
    nodes = df_full_nodes.drop_duplicates(keep='first')
    df_full_edges = pd.DataFrame.from_dict(new_dict_edges)
    return_dict['nodes'] = nodes
    return_dict['edges'] = df_full_edges
    
    return return_dict

'''
    networks_controller: Uses Dict data structure hydrated from the 
        following functions
            - .batch_map()
            - .ftree_edge_maker(), and
            - .infomap_hub_maker().
        It appends node names to edge data and also creates a node list
        for each module.
        Args:
            - p_sample: Tuple of Integers. Number of desired period range to sample.
            - m_sample: Tuple of Integers. Number of desired module range to sample.
                - These tuples assume continuous ranges: 1-10, 3-6, etc.
            - dict_im: Dict. Output from batch_map(), ftree_edge_maker(), and
                infomap_hub_maker(), which includes.
                - DataFrame. Module edge data.
                - List of dicts. Module node data with names.
        Return:
            - dict_network: Appends more accessible edge and node data.
'''
def networks_controller(p_sample, m_sample, dict_im):
    dict_network = {}
    dict_network['network'] = {}
    for p in range(p_sample[0], (p_sample[1]+1)):
        print('Processing period', p)
        dict_network['network'][str(p)] = {}
        for m in range(m_sample[0], (m_sample[1]+1)):
            print('Module', m)
            # Check if module exists
            if str(m) in dict_im[str(p)]['indices']['ftree_links']:
                # Check if dataframe exists
                if 'df_edges' in dict_im[str(p)]['indices']['ftree_links'][str(m)]:
                    nodes_and_edges = network_organizer(
                                dict_im[str(p)]['indices']['ftree_links'][str(m)]['df_edges'],
                                dict_im[str(p)]['info_hub'][str(m)]
                    )
                    dict_network['network'][str(p)][str(m)] = nodes_and_edges
            
    return dict_network

'''
    append_percentages: Helper function for ranker(). Appends each node's total_percentage to the list
    - Args:
        - rl= List of lists. Ranked list of nodes per hub
'''
def append_percentages(rl):
    for n in rl:
        if n[3] == 1:
            n.append(1.0)
        else:
            percent = float(n[1]) / float(n[2])
            n.append(float(percent))
    return rl
    
'''
    append_rank: Helper function for ranker(). It appends the rank number for the 'spot' value.
'''
def append_rank(lnr):
    i = 1
    for n in lnr:
        n.append(i)
        i = i +1
    return lnr

'''
    ranker: Appends flow rank and percentages at different aggregate levels.
    - Args:
        - rank_type= String. Argument option for type of ranking to conduct. Currently only per_hub.
        - tdhn= Dict of corpus. Traverses the 'info_hub'
    - Return
        - tdhn= Updated 'info_hub' with 'percentage_total' per hub and 'spot' for each node per hub,
    - TODO: Add per_party and per_hubname
'''
def ranker(tdhn, **kwargs):
    if kwargs['rank_type'] == 'per_hub':
        for p in tdhn:
            if 'info_hub' in tdhn[p]:
                for h in tdhn[p]['info_hub']:
                    # Write list of hub's nodes and flow scores as a list of tuples
                    list_nodes_unranked = []
                    for n in tdhn[p]['info_hub'][h]:
                        if 'total_hub_flow_score' in n:
                            name = n['name']
                            flow = n['score']
                            total_hub_flow = n['total_hub_flow_score']
                            listed_rank = [ name, format(flow, 'f'), format(total_hub_flow, 'f') ]
                            list_nodes_unranked.append(listed_rank)
                    # Rank the list
                    list_nodes_ranked = sorted(list_nodes_unranked, key=lambda x: x[1], reverse=True)
                    list_nodes_appended_ranks = append_rank(list_nodes_ranked)
                    listed_nodes_percentages = append_percentages(list_nodes_appended_ranks)

                    # Update hub Dict with percentages
                    for per in tdhn[p]['info_hub'][h]:
                        for lnp in listed_nodes_percentages:
                            if per['name'] == lnp[0]:
                                per.update({'percentage_total': lnp[4]})
                                per.update({'spot': lnp[3]})
        return tdhn

'''
    infomap_hub_maker: Takes fully hydrated Dict of the map or ftree files
        and parses its Nodes into per Period and Module Dicts.
        - Args: 
            - file_type= String. 'map' or 'ftree' file type designation
            - dict_map= Dict of map files
            - mod_sample_size= Integer. Number of modules to sample
            - hub_sample_size= Integer. number of nodes to sample for "hub" of each module.
                Or -1 to capture all nodes in hub.
        - Output:
            - dict_map= Dict with new 'info_hub' key hydrated with hubs
'''
def infomap_hub_maker(dict_map, **kwargs):
    re_mod = r"\d{1,}\:"        #Map/Ftree module number
    re_node = r"\:\d{1,}"      #Map/Ftree node number
    re_name = r"\"\S{1,}\""     #Map node name
    re_score = r"\d{1}\.\d{1,}" #Node's flow score
    
    re_ftree_module = r"\d{1,2}\:"
    re_ftree_flow = r"((\d\.\d{1,}\s)|(\d\.\d{1,}e\-\d{0,}\s)|(\s0\s\"))"
    re_ftree_username = r"\".{0,}\""
    re_ftree_node = r"\"\s\d{0,}"
    
    # Traverse each period in dictionary
    for p in dict_map:
        dict_hubs = {}
        
        if kwargs['file_type'] == 'map':
            # Get indices for nodes in period's map
            start = dict_map[p]['indices']['nodes'][0]
            end = dict_map[p]['indices']['nodes'][1]
            
            # Traverse nodes in period's map
            for n in dict_map[p]['lines'][start:end]:
                mod_match = re.match(re_mod, n)
                node_match = re.findall(re_node, n)
                name_match = re.findall(re_name, n)
                score_match = re.findall(re_score, n)

                mod = mod_match[0][0:-1]
                node = node_match[0][1:]
                name = name_match[0][1:-1]
                score = score_match[0]
                score = float(score)
                hub_list = []

                # Retrieve first n modules
                if int(mod) == (kwargs['mod_sample_size']+1):
                    # Update period with infomap hubs
                    dict_map[p].update({
                        'info_hub': dict_hubs
                    })
                    break # All done
                elif mod not in dict_hubs:
                    dict_hubs.update({mod:
                        [{
                            'node': node,
                            'name': name,
                            'score': score
                         }]
                    })
                elif mod in dict_hubs and len(dict_hubs[mod]) >= 1 and (len(dict_hubs[mod]) < kwargs['hub_sample_size'] or kwargs['hub_sample_size'] == -1):
                    hub_list = {
                        'node': node,
                        'name': name,
                        'score': score
                    }
                    dict_hubs[mod].append(hub_list)
        if kwargs['file_type'] == 'ftree':
            # Get indices for ftree in period's flow values
            start = dict_map[p]['indices']['ftree_modules'][0]
            end = dict_map[p]['indices']['ftree_modules'][1]
            
            # Traverse nodes in period's ftree
            for n in dict_map[p]['lines'][start:end]:
                mod_match = re.match(re_ftree_module, n)
                flow_match = re.findall(re_ftree_flow, n)
                name_match = re.findall(re_ftree_username, n)
                node_match = re.findall(re_node, n)
                
                mod = mod_match[0][0:-1] #Remove colon
                flow = flow_match[0][0][:-1] #Remove whitespace at end
                flow = float(flow)
                name = name_match[0][1:-1] #Remove double-quotes
                node = node_match[0][1:] #Remove colon at start
                
                hub_list = []

                 # Retrieve first n modules
                if int(mod) == (kwargs['mod_sample_size']+1):
                    # Update period with infomap hubs
                    dict_map[p].update({
                        'info_hub': dict_hubs
                    })
                    break # All done
                elif mod not in dict_hubs and node != '0 "':
                    dict_hubs.update({mod:
                        [{
                            'module': mod,
                            'node': node,
                            'name': name,
                            'score': flow
                         }]
                    })
                elif mod in dict_hubs and len(dict_hubs[mod]) >= 1 and (len(dict_hubs[mod]) < kwargs['hub_sample_size'] or kwargs['hub_sample_size'] == -1) and node != '0 "':
                    hub_list = {
                        'module': mod,
                        'node': node,
                        'name': name,
                        'score': flow
                    }
                    dict_hubs[mod].append(hub_list)
        
    return dict_map

'''
    get_score_total: Helper function for score_summer. Tallies scores per Hub.
    - Args:
        - list_nodes= List of Dicts
    - Return
        - total= Float. Total flow score for a Hub.
'''
def get_score_total(list_nodes):
    total = 0.0
    for n in list_nodes:
        s = format(n['score'], 'f')
        total = total + float(s)
    return total

'''
  get_period_flow_total: Helper function for score_summer. Tallies scores per Period across hubs.
  - Args:
      - lpt= List. Contains hub totals per Period.
  - Return
      - Float. Total flow score for a Period. 
'''
def get_period_flow_total(lpt):
    pt = 0.0
    for t in lpt:
        pt = pt + t
    return pt 
    
'''
    score_summer(): Tally scores from each module per period and append a score_total to each node instance per module for every period.
    - Args:
        - dhn= Dict of hubs returned from infomap_hub_maker
'''
def score_summer(dhn, **kwargs):
    for p in dhn:
        list_period_totals = []
        if 'info_hub' in dhn[p]:
            for h in dhn[p]['info_hub']:
                total_hub_flow = 0.0
                total_hub_flow = get_score_total(dhn[p]['info_hub'][h])

                # Update hub with flow score total for each node in hub
                h_hub_sample = 0
                for ln in dhn[p]['info_hub'][h]:
                    if kwargs['hub_sample_size']:
                        if h_hub_sample < kwargs['hub_sample_size']:
                            ln.update({'total_hub_flow_score': total_hub_flow})
                            h_hub_sample = h_hub_sample+1
                    else:
                        ln.update({'total_hub_flow_score': total_hub_flow})

                # Append hub total to list for period tally later
                list_period_totals.append(total_hub_flow)
        pt = get_period_flow_total(list_period_totals)
        # Update each node in each hub in each period with period flow scores
        if 'info_hub' in dhn[p]:
            for ht in dhn[p]['info_hub']:
                # Update hub with flow score total for each node in hub
                ht_hub_sample = 0
                for pln in dhn[p]['info_hub'][ht]:
                    if kwargs['hub_sample_size']:
                        if ht_hub_sample < kwargs['hub_sample_size']:
                            pln.update({'total_period_flow_score': pt})
                            ht_hub_sample = ht_hub_sample+1
                    else:
                        pln.update({'total_period_flow_score': pt})
    return dhn

'''
    output_infomap_hub: Takes fully hydrated infomap dict and outputs it as a CSV file.
        - Args: 
            - header= column names for DataFrame and CSV; 
                - Assumes they're in order with period and hub in first and second position
            - dict_hub= Hydrated Dict of hubs
            - filtered_hub_length= Int. Desired length of hub
            - path= Output path
            - file= Output file name
'''
def output_infomap_hub(**kwargs):
    hubs = []
    for p in kwargs['dict_hub']:
        if 'info_hub' in kwargs['dict_hub'][p]:
            for h in kwargs['dict_hub'][p]['info_hub']:
                tracker = 0
                for r in kwargs['dict_hub'][p]['info_hub'][h]:
                    if tracker < kwargs['filtered_hub_length']:
                        temp_hub = [int(p)]
                        for c in r:
                            temp_hub.append(r[c])
                        hubs.append(temp_hub)
                        tracker = tracker+1
    df_info_hubs = pd.DataFrame(hubs, columns=kwargs['header'])

    df_info_hubs.to_csv(join(kwargs['path'], kwargs['file']),
                                    sep=',',
                                    encoding='utf-8',
                                    index=False)
    print(kwargs['file'], ' written to ', kwargs['path'])

'''
    add_infomap: Helper function for sampling_module_hubs. 
    - Args:
        - dft: DataFrame of sampled tweet data
        - dfh: Full DataFrame of hubs data
        - period_num: Integer of particular period number
        - hub_cols: List of column names from hub file desired to preserve
    - Returns List of Dicts with hub and info_name mentions info
'''
def add_infomap(dfh, dft, hub_user_col, corpus_user_col, period_col, hub_col):
    period_comms = []
    for mrow in dfh.to_dict('records'):
        # 1. Find user's tweets
        author_query = dft[dft[corpus_user_col] == mrow[hub_user_col]]

        # 2. Track period and hub
        current_period_num = mrow[period_col]
        current_hub_num = mrow[hub_col]

        current_hub = dfh[ (dfh[period_col] == current_period_num) & (dfh[hub_col] == current_hub_num) ]

        # 3. Find tweets with module mentions
        for trow in author_query.to_dict('records'):
            m = trow['mentions']
            if isinstance(m, str):
                m = ast.literal_eval(m)
                if len(m) > 0:
                    # 4. Go through list of mentions
                    for mentioned in m:
                        new_dict = {}
                        # 5. Go through hub list 
                        for module in current_hub.to_dict('records'):
                            # 6. If matched module, append
                            if mentioned == module[hub_user_col]:
                                new_dict.update({
                                    'period':current_period_num,
                                    'module':current_hub_num,
                                })
                                
                                for r in trow:
                                    new_dict.update({
                                        r: trow[r]
                                    })
                                
                                # 3. Append to main list
                                period_comms.append(new_dict)

                            # found_hubber = dfh[dfh[hub_user_col] == mentioned]
                            # fh = found_hubber.to_dict('records')
                            # if len(fh) > 0:
                            #     for i in fh:
                            #         new_dict = {}                            
                            #         # 1. Append desired hub columns
                            #         for c in hub_cols:
                            #             new_dict.update({
                            #                 c: i[c]
                            #             })
                                    
                            #         # 2. Append tweet data
                            #         for r in row:
                            #             new_dict.update({
                            #                 r: row[r]
                            #             })
                                    
                            #         # 3. Append to main list
                            #         period_comms.append(new_dict)
                    
    return period_comms

'''
    sampling_module_hubs: Compares hub set with tweet data to yield a
        sample of tweets with hub information. Sample meant to examine qualitative
        features of the modules per period.
        **Assumes you have sorted tweets in descending order by number of RTs.**
    - Args:
        - period_dates: Dict of lists that include dates for each period of the corpus
        - period_check: String for option: Check against 'single' or 'multiple'
        - period_num: Integer. If period_check == 'single', provide integer of period number.
        - df_all_tweets: Pandas DataFrame of tweets
        - df_hubs: Pandas DataFrame of infomapped hubs. Two required column names: 
            - period: Integer of particular period number
            - node_name: String. Represents target node to search.
        - top_rts_sample: Integer of desired sample size of sorted top tweets (descending order)
        - hub_sample: Integer of desired sample size to output.
        - hub_cols: List of column names from hub file desired to preserve.
    - Returns DataFrame of top sampled tweets 
'''
def sampling_module_hubs(**kwargs):
    module_output = []
    if kwargs['period_check'] == 'multiple':
        for p in kwargs['period_dates']:
            # 1. Use dates to find all tweets in period
            df_p = kwargs['df_all_tweets'][kwargs['df_all_tweets']['date'].isin(kwargs['period_dates'][p])]
            # 2. Sort tweets with top RT'd tweets in descending order
            df_top_rts = df_p.sort_values(['retweets_count'], ascending=[False])
            # 3. Save only top X sample
            df_output = df_top_rts[:kwargs['top_rts_sample']]
            
            print('Sample of top', 
                kwargs['top_rts_sample'],
                'RT\'d tweets written and sorted.\nNow writing new dataframe with combined hub information, which may take some time.')
            
            # 4. Send df_output, sliced_hub, and period_num to revised 
            sliced_hub = kwargs['df_hubs'][kwargs['df_hubs']['period'] == int(p)].copy()
            period_num = int(p)
            new_rows = add_infomap(
                dfh=sliced_hub,
                dft=df_output,
                hub_user_col=kwargs['hub_user_col'],
                corpus_user_col=kwargs['corpus_user_col'],
                period_col=kwargs['period_col'],
                hub_col=kwargs['hub_col']            )
            
            # 5. Append period's dicts of tweets to list
            module_output = module_output + new_rows
            print('Completed period', period_num, 'tweets.\n\n')
            
    elif kwargs['period_check'] == 'single':
        # 1. Use dates to find all tweets in period
        df_p = kwargs['df_all_tweets'][kwargs['df_all_tweets']['date'].isin(kwargs['period_dates'])]
        # 2. Sort tweets with top RT'd tweets in descending order
        df_top_rts = df_p.sort_values(['retweets_count'], ascending=[False])
        # 3. Save only top X sample
        df_output = df_top_rts[:kwargs['top_rts_sample']]
        
        print('Sample of top', 
            kwargs['top_rts_sample'],
            'RT\'d tweets written and sorted.\nNow writing new dataframe with combined hub information, which may take some time.')
        
        # 4. Send df_output, sliced_hub, and kwargs['period_num'] to revised 
        sliced_hub = kwargs['df_hubs'][kwargs['df_hubs']['period'] == kwargs['period_num']].copy()
        new_rows = add_infomap(
            dft=df_output,
            dfh=sliced_hub,
            period_num=kwargs['period_num']
        )

        module_output = module_output + new_rows
        print('Completed period', kwargs['period_num'], 'tweets.\n\n')
    
    module_df = pd.DataFrame(module_output)
    return module_df

'''
    batch_output_period_hub_samples: Periodic batch output that saves 
        sampled tweets as a CSV. Assumes successively numbered periods
    - Args:
        - module_output: DataFrame of tweet sample data per Period per Module
        - period_total: Interger of total number of periods
        - file_ext: String of desired filename extension pattern
        - period_path: String of desired path to save the files
    - Returns nothing
'''
def batch_output_period_hub_samples(**kwargs):
    x = 1
    while x <= kwargs['period_total']:
        p_file = 'p' + str(x) + kwargs['file_ext']
        output = kwargs['module_output'][ kwargs['module_output'].period == x ]
        output.to_csv(join(kwargs['period_path'], p_file), sep=',', encoding='utf-8', index=False)
        print('Wrote', p_file, 'to', kwargs['period_path'])
        x = x + 1

'''
    sample_getter: Samples corpus based on module edge data from infomap data.    
    Args:
        - sample_size: Integer. Number of edges to sample. To keep all results, use -1 (Int) value.
        - edges: List of Dicts. Edge data.
        - period_corpus: DataFrame. Content corpus to be sampled.
        - sample_type: String. Current options include:
            - 'modules': Samples tweets based on community module source-target relations.
            - 'ht_group': Samples tweets based on use of hashtags. Must provide list of strings.
        - user_threshold: Integer. If you want limit sampling an active user, set a limit.
        - random: Boolean. True will randomly sample fully retrieved set of tweet content
        - ht_list: List of strings. If sampling via hashtag groups, then provide a list of the hashtags. Default is None.
    Return:
        - DataFrame. Sampled content, based on infomap module edges.
'''
def sample_getter(sample_size, edges, period_corpus, sample_type, user_threshold, random, ht_list=None):
    mod_list_sample = []
    l = len(edges)
    for c in tqdm(range(0, l)):
        try:
            # Based on source - target and dates, search corpus for tweets as DF
            s = edges[c]['source_name']
            t = edges[c]['target_name']
            sample_content = period_corpus[period_corpus['username'] == s]
            if sample_type == 'modules':
                for row in sample_content.to_dict('records'):
                    m = row['mentions']
                    if isinstance(m, str):
                        m = ast.literal_eval(m)
                        if len(m) > 0:
                            # Go through list of mentions
                            for mentioned in m:
                                if mentioned == t:
                                    # Append to main list
                                    mod_list_sample.append(row)
            elif sample_type == 'hashtag_group':
                for row in sample_content.to_dict('records'):
                    m = row['hashtags']
                    if isinstance(m, str):
                        m = ast.literal_eval(m)
                        if len(m) > 0:
                            for i in m:
                                for h in ht_list:
                                    if i == h:
                                        mod_list_sample.append(row)
                                        
        except IndexError as e:
            print('\n\nERROR:',e, 'Out of edges to sample.\n\n')
            break
    # Remove duplicates
    result_list = [i for n, i in enumerate(mod_list_sample) if i not in mod_list_sample[n + 1:]]
    
    # Sort in Ascending order as per RT count
    new_result_list = sorted(result_list, key=lambda k: k['retweets_count'], reverse=True)
    res_list = []
    for s in new_result_list:
        if len(res_list) == 0:
            res_list.append(s)
        if len(res_list) > 0:
            # Check how many times user has been sampled
            user_check = 0
            for c in res_list:
                if c['username'] == s['username']:
                    user_check = user_check + 1
            # If specific user does not exceed given threshold, append tweet
            if user_check < user_threshold:
                res_list.append(s)
    
    # Add to sample
    if sample_size == -1:
        # Retrieve total sample
        df_sample = pd.DataFrame(res_list)
        return df_sample
    else:
        if len(res_list) > (sample_size):
            df_sample = pd.DataFrame(res_list)
            if random == True:
                # Sample it randomly
                random_sample = df_sample.sample(n=sample_size, random_state=1)
                return random_sample
            elif random == False:
                sorted_sample = df_sample.sort_values('retweets_count', ascending=False)
                ss = sorted_sample[:sample_size]
                return ss
        elif len(res_list) < sample_size:
            df_sample = pd.DataFrame(res_list)
            return df_sample
       
'''
    content_sampler: Sample content in each period per module, based on
        map equation flow-based community detection.
        Args:
            - period_type: String. Options include:
                - 'single': If sampling single period from dict with [module] structure
                - 'multiple': If sampling multiple periods from dict with [period][module] structure
            - network: Dict. Each community across periods edge and node data.
            - corpus: DataFrame.
            - sample_type: String. Current options include and passed onto ic_sample_getter():
                - 'modules': Samples tweets based on community module source-target relations.
                - 'ht_groups': Samples tweets based on use of hashtags. Must provide list of strings.
            - period_dates: Dict of lists.
            - sample_size: Integer. Number of edges to sample. To keep all results, use -1 (Int) value.
            - period_num: String. If single 'period_type', define period number as a string.
            - random: Boolean. True pulls randomized sample. False pulls top x tweets.
        Return:
            - Dict of DataFrames. Sample of content in each module per period       
'''
def content_sampler(period_type, network, sample_size, period_dates, corpus, sample_type, ht_group, user_threshold, period_num=None, random=False):
    dict_samples = {}
    # Single period
    if period_type == 'single':
        for m in network:
            print('Module', m, 'started.')
            dict_samples[m] = {}
            m_edges = network[m]['edges'].to_dict('records') #Dict of module edge data
            p_dates = period_dates[period_num] #List of dates for period
            p_corpus = corpus.loc[corpus['date'].isin(p_dates)]
            sample = sample_getter(
                                    sample_size, 
                                    m_edges, 
                                    p_corpus,
                                    sample_type,
                                    ht_list=ht_group,
                                    user_threshold=user_threshold,
                                    random=random)
            try:
                print('Module', m, 'sample size:', len(sample))
                dict_samples[m]['sample'] = sample
            except TypeError as e:
                print(e, 'Module', m, 'sample size: 0')
        
    elif period_type == 'multiple':
        # Multiple periods
        for p in network:
            dict_samples[p] = {}
            print('Sampling from period', p)
            for m in network[p]:
                print('Module', m, 'started.')
                dict_samples[p][m] = {}
                m_edges = network[p][m]['edges'].to_dict('records') #Dict of module edge data
                p_dates = period_dates[p] #List of dates for period
                p_corpus = corpus.loc[corpus['date'].isin(p_dates)]
                sample = sample_getter(
                                        sample_size, 
                                        m_edges, 
                                        p_corpus,
                                        sample_type,
                                        ht_list=ht_group,
                                        user_threshold=user_threshold,
                                        random=random)
                try:
                    print('Module', m, 'sample size:', len(sample))
                    dict_samples[p][m]['sample'] = sample
                except TypeError as e:
                    print(e, 'Module', m, 'sample size: 0')

            print('\n')
                
    return dict_samples

'''
    edges_sampler: Sample edges in each period per module, based on
        map equation flow-based community detection.
        Args:
            - network: Dict. Each module edges data across periods edge and node data.
            - sample_size: Integer.
            - column_name: String. Name of desired column.
            - random: Boolean. True pulls randomized sample. False pulls top x tweets.
        Return:
            - Dict of DataFrames. Sample of content in each module per period       
'''
def edges_sampler(network, sample_size, column_name, random=False):
    dict_samples = {}
    if random == False:
        for p in network:
            dict_samples[p] = {}
            print('Sampling from period', p)
            for m in network[p]:
                dict_samples[p][m] = {}
                # Remove duplicate usernames, before sampling
                no_dupes = network[p][m]['edges'].drop_duplicates(subset=column_name,keep='first')
                
                ss_length = len(no_dupes)
                sample_col_name = 'sample_'+column_name
                if ss_length < sample_size:
                    df_sample = no_dupes[:ss_length]
                    dict_samples[p][m][sample_col_name] = df_sample[column_name]
                if ss_length > sample_size:
                    df_sample = no_dupes[:sample_size]
                    dict_samples[p][m][sample_col_name] = df_sample[column_name]
        return dict_samples
    if random == True:
        for p in network:
            dict_samples[p] = {}
            print('Sampling from period', p)
            for m in network[p]:
                dict_samples[p][m] = {}
                # Remove duplicate usernames, before sampling
                no_dupes = network[p][m]['edges'].drop_duplicates(subset=column_name,keep='first')
                
                ss_length = len(no_dupes)
                sample_col_name = 'sample_'+column_name
                if ss_length < sample_size:
                    df_sample = no_dupes.sample(n=ss_length, random_state=1)
                    dict_samples[p][m][sample_col_name] = df_sample[column_name]
                if ss_length > sample_size:
                    df_sample = no_dupes.sample(n=sample_size, random_state=1)
                    dict_samples[p][m][sample_col_name] = df_sample[column_name]
        return dict_samples

##################################################################

## allPeriodsObject Functions

##################################################################
'''
'''
def period_maker(bd, ed):
    # Make period date-range
    begin_date = arrow.get(bd, 'YYYY-MM-DD')
    end_date = arrow.get(ed, 'YYYY-MM-DD')
    date_range = arrow.Arrow.range('day', begin_date, end_date)
    return date_range

'''
    period_writer():  Accepts list of lists of period date information
    and returns a Dict of per Period dates for temporal analyses.
        - Args:
            - periodObj: Optional first argument periodObject, Default is None
            - 'ranges': Hierarchical list in following structure:
                ranges = [
                    ['p1', ['2018-01-01', '2018-03-30']],
                    ['p2', ['2018-04-01', '2018-06-12']],
                    ...
                ]
'''
def period_dates_writer(allPeriodsObj=None, **kwargs):
    period_dict = {}
    for r in kwargs['ranges']:
        period_list = []
        p_dates = period_maker(r[1][0], r[1][1]) # send period date range
        for d in p_dates:
            # Append returned date range to period list
            period_list.append( str(d.format('YYYY-MM-DD')) )
        period_dict.update({r[0]: period_list})

    if allPeriodsObj == None:
        return period_dict
    else:
        allPeriodsObj.period_dates = period_dict
        return allPeriodsObj

'''
    get_comm_nums(): Filters community column values into List
        Args: 
            - period_obj= Instantiated periodObject()
            - dft_comm_col= Dataframe column of community values of nodes
        Returns: A List of unique community numbers (Strings) within the period
            - Either the periodObject() with the new property comm_nums, or
            - List of comm numbers as Strings
'''
def get_comm_nums(dft_comm_col=None,period_obj=None):
    # Get community numbers
    c_list = []
    for c in dft_comm_col.values.tolist():
        if not c_list:
            c_list.append(c)
        elif c not in c_list:
            c_list.append(c)
    
    if period_obj is None:
        return c_list
    elif period_obj:
        period_obj.comm_nums = c_list
        return period_obj

'''
    comm_sender && write_community_list functions create a dict of nodes 
    and edges to be saved as a property, .subgraphs_dict, of a periodObject.

    1. Creates a List of nodes per Community
    2. Creates a List of edges per Community
    3. Appends dict of these lists to comprehensive dict for the period.
    4. Appends this period dict to the period)bject property: .subgraphs_dict
    5. Returns the object.

    Args: 
        - comm_nums= List of community numbers from periodObject.comm_nums
        - period_obj= Instantiated periodObject()
        - nodes= Dataframe of community nodes with a column named 'community'
        - edges= Dataframe of community edges
    Returns:
        - periodObject() with the new property .subgraphs_dict
'''
def comm_sender(**kwargs):
    new_comm_dict = {}
    for a in kwargs['comm_nums']:
        cl = []
        cl = [a]
        print(cl)
        comm_nodes = pd.DataFrame()
        comm_nodes = kwargs['nodes'][kwargs['nodes'].community.isin(cl)]
        parsed_comm = write_community_list(comm_nodes, kwargs['edges'], a)
        new_comm_dict.update(parsed_comm)
    print(len(new_comm_dict))
    kwargs['period_obj'].subgraphs_dict = new_comm_dict
    return kwargs['period_obj']

def write_community_list(cn, df_edges, a):
    node_list = []
    edge_list = []
    
    for n in cn.values.tolist():
        node_list.append(n[0])
    
    print('Sample NODES: ', node_list[:5])
    
    for e in df_edges.values.tolist():
        # ONLY Comm SRC-->TARGETS
        for c in node_list:
            if e[0] == c:
                for cc in node_list:
                    if e[1] == cc:
                        edge_list.append( (e[0], e[1]) )
    
    print('Sample EDGES: ', edge_list[:5])
    dict_comm = {}
    dict_comm.update({a: { 'nodes': node_list, 'edges': edge_list }})
    return dict_comm

'''
    add_comm_nodes_edges(): Function to more quickly generate new networkX graph of
        specific comms in a period
    Args: 
        - Nodes: 
        - Newly instantiated networkX graph object
        - Edges: 
    Returns: networkX graph object with nodes and edges
'''
def add_comm_nodes_edges(**kwargs):
    for n in kwargs['comms_dict']['nodes']:
        kwargs['g'].add_node(n)
    for source, target in kwargs['comms_dict']['edges']:
        kwargs['g'].add_edge(source, target)
    return kwargs['g']

'''
    add_all_nodes_edges(): Function to more quickly generate new 
        networkX graph of all comms in a period
    Args: 
        - Nodes: 
        - Newly instantiated networkX graph object
        - Edges: 
    Returns: networkX graph object with nodes and edges
'''
def add_all_nodes_edges(**kwargs):
    for cd in kwargs['comms_dict']:
        for n in kwargs['comms_dict'][cd]['nodes']:
            kwargs['g'].add_node(n)
        for source, target in kwargs['comms_dict'][cd]['edges']:
            kwargs['g'].add_edge(source, target)
    return kwargs['g']

'''
    Draws subgraphs with networkX module
    Args:
        plt.figure():
            - figsize= Tuple of (width,height) Integers, e.g., (50,35) for matplot figure
        nx.draw():
            - with_labels= Boolean for labels option
            - font_weight= value for networkX option (see spec)
            - node_size= value for networkX option (see spec)
            - width: value for networkX option for edge width
        nx.draw_networkx_nodes() and nx.draw_networkx_edges():
            - period_comm_list= #List of tuples, where the first value
                is the period object, and the second the specified community:
                [(p1_obj, 8), (p2_obj, 18)]
            - node_size= #Integer
            - edge_color= #Hex color code
            - edge_width= #Integer
            - edge_alpha= #Float b/t 0 and 1
            - axis= #string 'on' or 'off' value
            - graph_titles= #List of Strings of desired titles for each 
                graph. Its order should follow period_comm_list[]
            - font_dict= #Dict with font options via matplot spec
            - output_paths= #List of Strings of desired paths and filenames 
                to save the image. Its order should follow period_comm_list[].
'''
def draw_subgraphs(**kwargs):
    period_index = 0
    # For each community list, draw the nodes and edges 
    # with specifying attributes
    for pc in kwargs['period_comm_list']:
        plt.figure(figsize=kwargs['figsize'])
        G = nx.DiGraph()
        if len(list(G.nodes())) > 0:
            G.clear() #Fresh graph
        else:
            G = add_comm_nodes_edges(comms_dict=pc[0].subgraphs_dict[pc[1]], g=G)
            pos_custom = nx.nx_agraph.graphviz_layout(G, prog=kwargs['graph_type'])

            nx.draw(G, pos_custom, with_labels=kwargs['with_labels'],
                font_weight=kwargs['font_weight'], node_size=kwargs['node_size'], width=kwargs['width'])

            node_list = pc[0].subgraphs_dict[pc[1]]['nodes']
            edge_list = pc[0].subgraphs_dict[pc[1]]['edges']
            # Draw nodes
            nx.draw_networkx_nodes(
                G,
                pos_custom,
                nodelist=node_list,
                #update existing dict, before calling this func
                node_color=pc[0].subgraphs_dict[pc[1]]['node_color'],
                node_size=kwargs['node_size'])
            # Draw edges
            nx.draw_networkx_edges(
                G,
                pos_custom,
                edgelist=edge_list,
                #update existing dict, before calling this func
                edge_color=pc[0].subgraphs_dict[pc[1]]['edge_color'],
                width=kwargs['edge_width'],
                alpha=kwargs['edge_alpha'])

            plt.axis(kwargs['axis'])
            plt.title(kwargs['graph_titles'][period_index], fontdict=kwargs['font_dict'])
            plt.savefig(kwargs['output_paths'][period_index]) # save as image
            plt.show()
            period_index = period_index + 1

##################################################################

## communitiesObject Functions

##################################################################

'''
    Writes all objects and their respective source/target information
    to a CSV of "hubs"
'''
def create_hub_csv_files(**kwargs):
    list_of_dfs = []
    for fb in kwargs['full_obj']:
        list_of_dfs.append(kwargs['full_obj'][fb].full_hub)
    df_merged = pd.concat(list_of_dfs, axis=0).reset_index(drop=True)
    if kwargs['drop_dup_cols'] == True:
        df_merged_drop_dup = df_merged.loc[:,~df_merged.columns.duplicated()]
        write_csv(df_merged_drop_dup, kwargs['sys_path'], kwargs['output_file'])
    else:
        write_csv(df_merged_drop_dup, kwargs['sys_path'], kwargs['output_file'])
    print('File written to system.')

'''
    get_all_comms: Slice the full set to community and their respective tweets. 
    - Args: 
        - dft: Dataframe
        - col_community: String. Column name for community
        - col_tweets: String. Column name for tweet content
'''
def get_all_comms(dft, col_community, col_tweets):
    new = []
    for row in dft.to_dict('records'):
        r = {
            col_community: row[col_community],
            col_tweets: row[col_tweets]
        }
        new.append(r)
    df = pd.DataFrame(new)
    return df

'''
    Write per Community content segments into a dictionary. This requires a corpus with
        defined period and community module columns.
    - Args:
        - comm_list= List of community numbers / labels
        - df_content= DataFrame of data set in question
        - comm_col= String of column name for community/module
        - content_col= Sring of column name for content to parse and examine
        - sample_size_percentage= Desired percentage to sample from full set
    - Returns Dict of sliced DataFrames (value) as per their community/module (key)
'''
def comm_dict_writer(**kwargs):
    dict_c = {}
    all_comms = kwargs['df_content']
    for c in kwargs['comm_list']:
        out_comm_obj = communitiesObject()
        all_comms = get_all_comms(kwargs['df_content'], kwargs['comm_col'], kwargs['content_col'])
        df_slice = all_comms[all_comms[kwargs['comm_col']] == c]
        df_slice = df_slice.reset_index(drop=True)
        out_comm_obj.content_slice = df_slice
        dict_c.update( { str(c): out_comm_obj } )

    return dict_c

'''
    Isolates community's content, then splits string into list of strings per Tweet
        preparing them for the topic modeling.
        Args: 
            - single: Boolean. True equals single sample vs. multiple communities
            - col_name: String. Community label as String, 
            - dict_comm_obj: Dict of community objects
            - sample_size_percentage: Float. Between 0 and 1. 
            - stop_words: Stop words list of strings
        Returns as Dataframe of content for respective community
'''
def split_community_tweets(**kwargs):
    if kwargs['single'] == False:
        for cdf in kwargs['dict_comm_obj']:
            # Sample size
            print('Length of community', cdf, 'data set:', len(kwargs['dict_comm_obj'][cdf].content_slice))
            sample_size = len(kwargs['dict_comm_obj'][cdf].content_slice) * kwargs['sample_size_percentage']
            print('Sample size: ', int(sample_size))

            c_content = kwargs['dict_comm_obj'][cdf].content_slice[kwargs['col_name']][:int(sample_size)]
            # Split content segments; includes emoji support
            c_split = []
            for t in c_content.values.tolist():
                em_split_emoji = emoji.get_emoji_regexp().split(t)
                em_split_whitespace = [substr.split() for substr in em_split_emoji]
                em_split = functools.reduce(operator.concat, em_split_whitespace)
                # Append split content to list
                c_split.append(em_split)

            # Transform list into Dataframe
            df_documents = pd.DataFrame()
            for ts in c_split:
                df_documents = df_documents.append( {kwargs['col_name']: ts}, ignore_index=True )

            # Transform into list of processed docs
            split_docs = df_documents[kwargs['col_name']]
            cleaned_split_docs = clean_split_docs(split_docs, kwargs['stop_words'])
            kwargs['dict_comm_obj'][cdf].split_docs = cleaned_split_docs

    elif kwargs['single'] == True:
        cdf = kwargs['dict_comm_obj'].content_slice
        
        # Sample size
        print('Length of community data set:', len(cdf))
        
        sample_size = len(cdf) * kwargs['sample_size_percentage']
        
        print('Sample size: ', int(sample_size))

        c_content = cdf[kwargs['col_name']][:int(sample_size)]
        
        # Split content segments; includes emoji support
        c_split = []
        for t in c_content.values.tolist():
            em_split_emoji = emoji.get_emoji_regexp().split(t)
            em_split_whitespace = [substr.split() for substr in em_split_emoji]
            em_split = functools.reduce(operator.concat, em_split_whitespace)
            # Append split content to list
            c_split.append(em_split)

        # Transform list into Dataframe
        df_documents = pd.DataFrame()
        for ts in c_split:
            df_documents = df_documents.append( {kwargs['col_name']: ts}, ignore_index=True )

        # Transform into list of processed docs
        split_docs = df_documents[kwargs['col_name']]
        cleaned_split_docs = clean_split_docs(split_docs, kwargs['stop_words'])
        kwargs['dict_comm_obj'].split_docs = cleaned_split_docs
    
    print( ' \'processed_docs\': dataframe written for each community dictionary.' )
    return kwargs['dict_comm_obj']

'''
    Removes punctuation, makes lowercase, removes stopwords, and converts into dataframe for topic modeling
'''
def clean_split_docs(pcpd, stop):
    # Remove punctuation
    translator = str.maketrans('', '', string.punctuation)
    c_no_puncs = []
    for st in pcpd.values.tolist():
        s_list = []
        for s in st:
            # pass the translator to the string's translate method
            s_list.append( s.translate(translator) )
        c_no_puncs.append(s_list)

    # Make all lowercase
    c_no_puncs_lower = []
    for cc in c_no_puncs:
        new_list = []
        for c in cc:
            new_str = c.lower()
            if new_str:
                new_list.append(new_str)
        c_no_puncs_lower.append(new_list)

    # Remove stopwords
    c_cleaned = []
    for a in c_no_puncs_lower:
        ctw = []
        for st in a:
            if st not in stop:
                ctw.append(st)
        c_cleaned.append(ctw)

    # Convert list to dataframe
    df_c_cleaned_up = pd.DataFrame({ 'tweet': c_cleaned })
    p_clean_docs = df_c_cleaned_up['tweet']

    return p_clean_docs

'''
    tm_maker: Create data for TM.
    - Args: Pass many of the gensim LDATopicModel() object arguments here, plus some helpers. See their documentation for more details (https://radimrehurek.com/gensim/models/ldamodel.html).
        - random_seed: Integer. Value for randomized seed.
        - single: Boolean. True assumes only one period of data being evaluated.
        - split_comms: 
            - If 'single' False, Dict of objects with respective TM data.
            - If 'single' True, object with TM data
        - num_topics: Integer. Number of topics to produce (k value)
        - random_state: Integer. Introduce random runs.
        - update_every: Integer. "Number of documents to be iterated through for each update. Set to 0 for batch learning, > 1 for online iterative learning."
        - chunksize: Integer. "Number of documents to be used in each training chunk."
        - passes: Integer. "Number of passes through the corpus during training."
        - alpha: String. Pass options available via gensim package
        - per_word_topics: Boolean. 
    - Returns: Either updated Dict of objects, or single Dict. Now ready for visualization or printing.
'''
def tm_maker(**kwargs):
    np.random.seed(kwargs['random_seed'])
    nltk.download('wordnet')
    if kwargs['single'] == True:
        # Create Dictionary
        id2word = corpora.Dictionary(kwargs['split_comms'].split_docs.tolist())
        kwargs['split_comms'].id2word = id2word

        # Create Texts
        texts = kwargs['split_comms'].split_docs.tolist()
        kwargs['split_comms'].texts = texts

        # Term Document Frequency
        corpus = [id2word.doc2bow(text) for text in texts]
        kwargs['split_comms'].corpus = corpus

        # Human readable format of corpus (term-frequency)
        read_me = [[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]]
        kwargs['split_comms'].readme = read_me

        # Build LDA model
        lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=id2word,
                                                num_topics=kwargs['num_topics'],
                                                random_state=kwargs['random_state'],
                                                update_every=kwargs['update_every'],
                                                chunksize=kwargs['chunksize'],
                                                passes=kwargs['passes'],
                                                alpha=kwargs['alpha'],
                                                per_word_topics=kwargs['per_word_topics'])

        kwargs['split_comms'].model = lda_model

        # Compute Perplexity
        perplexity = lda_model.log_perplexity(corpus)
        kwargs['split_comms'].perplexity = perplexity
        print('\nPerplexity: ', perplexity)  # a measure of how good the model is. lower the better.

        # Compute Coherence Score
        coherence_model_lda = CoherenceModel(model=lda_model, texts=texts, dictionary=id2word, coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        kwargs['split_comms'].coherence = coherence_lda
        print('\nCoherence Score: ', coherence_lda)
    else:
        for split in kwargs['split_comms']:
            # Create Dictionary
            id2word = corpora.Dictionary(kwargs['split_comms'][split].split_docs.tolist())
            kwargs['split_comms'][split].id2word = id2word

            # Create Texts
            texts = kwargs['split_comms'][split].split_docs.tolist()
            kwargs['split_comms'][split].texts = texts

            # Term Document Frequency
            corpus = [id2word.doc2bow(text) for text in texts]
            kwargs['split_comms'][split].corpus = corpus

            # Human readable format of corpus (term-frequency)
            read_me = [[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]]
            kwargs['split_comms'][split].readme = read_me

            # Build LDA model
            lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                    id2word=id2word,
                                                    num_topics=kwargs['num_topics'],
                                                    random_state=kwargs['random_state'],
                                                    update_every=kwargs['update_every'],
                                                    chunksize=kwargs['chunksize'],
                                                    passes=kwargs['passes'],
                                                    alpha=kwargs['alpha'],
                                                    per_word_topics=kwargs['per_word_topics'])

            kwargs['split_comms'][split].model = lda_model

            # Compute Perplexity
            perplexity = lda_model.log_perplexity(corpus)
            kwargs['split_comms'][split].perplexity = perplexity
            print('\n', split, ' Perplexity: ', perplexity)  # a measure of how good the model is. lower the better.

            # Compute Coherence Score
            coherence_model_lda = CoherenceModel(model=lda_model, texts=texts, dictionary=id2word, coherence='c_v')
            coherence_lda = coherence_model_lda.get_coherence()
            kwargs['split_comms'][split].coherence = coherence_lda
            print('\n', split, ' Coherence Score: ', coherence_lda)

    print('\n Modeling complete.')
    return kwargs['split_comms']

'''
    Appends hubs' top RT'd tweets and users to respective period and community object
        -- Args:
            Dataframe of hub targets,
            Dict of Objects with .sources,
            String of period number
        -- Returns: Dict Object with new .top_rts per Object
'''
def get_hubs_top_rts(**kwargs):
    dft = kwargs['dft']
    all_tweets = dft.loc[:, lambda dft: ['community', 'username', 'tweets', 'retweets_count']]
    for tfd in kwargs['tdo']:
        per_comm_rts = all_tweets[all_tweets['community'] == tfd]
        per_comm_rts.rename(columns={'username': 'top_rters'}, inplace=True)
        top_renamed_per_comm_rts = per_comm_rts[:kwargs['top_num']].reset_index(drop=True)
        kwargs['tdo'][tfd].top_rts = top_renamed_per_comm_rts

    return kwargs['tdo']

'''
    Appends hubs' top mentions data to respective period and community object
        -- Args:
            Dataframe of hub top mentions,
            Dict of Objects,
            String of column name for period,
            String of period number,
            String of column name for the community number
        -- Returns: Dict Object with new .top_mentions per Object
'''
def get_hubs_top_mentions(**kwargs):
    for f in kwargs['dict_obj']:
        hub_comm = kwargs['hubs'][(kwargs['hubs'][kwargs['col_period']] == kwargs['pn']) & (kwargs['hubs'][kwargs['col_comm']] == f)]
        hub_comm = hub_comm.reset_index(drop=True)
        hub_comm.rename(columns={'username': 'top_mentions'}, inplace=True)
        kwargs['dict_obj'][f].top_mentions = hub_comm
    return kwargs['dict_obj']

'''
    Merges hubs' top RTs and targets data as a full list per Community
        -- Args:
        -- Returns:
'''
def merge_rts_mentions(fo):
    for f in fo:
        dfs = [df for df in [fo[f].top_mentions, fo[f].top_rts]]
        df_merged = pd.concat(dfs, axis=1).reset_index(drop=True)
        fo[f].full_hub = df_merged.reset_index(drop=True)
    return fo

'''
    print_keywords: Takes in a saturated dict of communities
        and their topic model component data (tmsd), then pretty
        prints it out to the terminal/notebook for quick access.
'''
def print_keywords(**kwargs):
    if kwargs['single'] == True:
        pprint(kwargs['tmsd'].model.print_topics())
    elif kwargs['single'] == False:
        for c in kwargs['tmsd']:
            print('COMMUNITY', c)
            pprint(kwargs['tmsd'][c].model.print_topics())
            print('\n\n')

##################################################################

## communityGroupsObject Functions

##################################################################
'''
    Processes input dataframe of network community hubs for use in the tsm.match_communities() function
        -- Args: A dataframe with Period, Period_Community (1_0), and top mentioned (highest in-degree) users
        -- Returns: Dictionary of per Period with per Period_Comm hub values as lists:
            {'1': {'1_0': ['nancypelosi','chuckschumer','senfeinstein',
                'kamalaharris','barackobama','senwarren','hillaryclinton',
                'senkamalaharris','repadamschiff','corybooker'],
               ...
               },
               ...
               '10': {'10_3': [...] }
            }
'''
def matching_dict_processor(**kwargs):
    full_dict = {}
    for index, row in kwargs['df'].iterrows():
        period_check = kwargs['df'].values[index][0]
        key_check = kwargs['df'].values[index][1]
        if index > 0 and (period_check in full_dict):
            if (kwargs['df'].values[index-1][1] == key_check) and (kwargs['df'].values[index-1][0] == period_check) and (key_check in full_dict[period_check]):
                # Update to full_dict[period_check][key_check]
                if kwargs['top_mentions'] == True:
                    full_dict[period_check][key_check].append(kwargs['df'].iloc[index]['top_mentions'])
                elif kwargs['top_rters'] == True:
                    full_dict[period_check][key_check].append(kwargs['df'].iloc[index]['top_rters'])
            elif  (kwargs['df'].values[index-1][1] == key_check) and (key_check not in full_dict[period_check]):
                # Create new key-value and update to full_dict
                if kwargs['top_mentions'] == True:
                    full_dict[period_check].update( { key_check: [kwargs['df'].iloc[index]['top_mentions']] } )
                elif kwargs['top_rters'] == True:
                    full_dict[period_check].update( { key_check: [kwargs['df'].iloc[index]['top_rters']] } )
        elif index > 0 and (period_check not in full_dict):
            if kwargs['top_mentions'] == True:
                full_dict.update( {period_check: { key_check: [kwargs['df'].iloc[index]['top_mentions']] } } )
            elif kwargs['top_rters'] == True:
                full_dict.update( {period_check: { key_check: [kwargs['df'].iloc[index]['top_rters']] } } )
        elif index == 0:
            if kwargs['top_mentions'] == True:
                full_dict.update( {period_check: { key_check: [kwargs['df'].iloc[index]['top_mentions']] } } )
            elif kwargs['top_rters'] == True:
                full_dict.update( {period_check: { key_check: [kwargs['df'].iloc[index]['top_rters']] } } )

    if kwargs['match_obj'] is None:
        return full_dict
    elif kwargs['match_obj'] is not None:
        if kwargs['top_mentions'] == True:
            kwargs['match_obj'].best_matches_mentions = full_dict
        elif kwargs['top_rters'] == True:
            kwargs['match_obj'].best_matches_rters = full_dict
        return kwargs['match_obj']

'''
    Takes period dict from matching_dict_processor() and submits to tsm.match_communities() method.
    Assigns, filters, and sorts the returned values into a list or tuples with findings.
        -- Args: Dictionary of per Period with per Period_Comm hub values as lists; filter_jacc threshold value (float) between 0 and 1.
        -- Returns: List of tuples: period_communityxperiod_community, JACC score
            [('1_0x4_0', 0.4286),
            ('1_0x2_11', 0.4615),
            ('1_0x3_5', 0.4615),
            ... ]
'''
def match_maker(**kwargs):
    pc_matching = {} # Assign with complete best matches
    for f1 in kwargs['full_dict']:
        for f2 in kwargs['full_dict']:
            if f1 != f2:
                # Runs similarity index (Jaccard's Co-efficient) on all period-comm combinations
                match = tsm.match_communities(kwargs['full_dict'][f1], kwargs['full_dict'][f2], weight_edges=False)
                the_key = f1 + 'x' + f2
                pc_matching.update({ the_key: match.best_matches })

    all_comm_scores = []
    for bmd in pc_matching:
        for b in pc_matching[bmd]:
            all_comm_scores.append( (b, pc_matching[bmd][b]) )

    sorted_all_comm_scores = sorted(all_comm_scores, key=lambda x: x[1])

    # Filter out low scores
    filtered_comm_scores = []
    for s in sorted_all_comm_scores:
        if s[1] > kwargs['filter_jacc']:
            filtered_comm_scores.append(s)

    sorted_filtered_comm_scores = sorted(filtered_comm_scores, key=lambda x: x[0][0], reverse=False)

    if kwargs['match_obj'] is None:
        return sorted_filtered_comm_scores
    elif kwargs['match_obj'] is not None:
        if kwargs['top_mentions'] is True:
            kwargs['match_obj'].sorted_filtered_mentions = sorted_filtered_comm_scores
            return kwargs['match_obj']
        elif kwargs['top_rters'] is True:
            kwargs['match_obj'].sorted_filtered_rters = sorted_filtered_comm_scores
            return kwargs['match_obj']

'''
    Plot the community comparisons as a bar chart
    -- Args:
        ax=None # Resets the chart
        counter = List of tuples returned from match_maker(),
        path = String of desired path to directory,
        output = String value of desired file name (.png)
    - Returns: Nothing.

'''
def plot_bar_from_counter(**kwargs):
    if kwargs['ax'] is None:
        fig = plt.figure()
        kwargs['ax'] = fig.add_subplot(111)

    frequencies = []
    names = []

    for c in kwargs['counter']:
        frequencies.append(c[1])
        names.append(c[0])

    N = len(names)
    x_coordinates = np.arange(len(kwargs['counter']))
    kwargs['ax'].bar(x_coordinates, frequencies, align='center')

    kwargs['ax'].xaxis.set_major_locator(plt.FixedLocator(x_coordinates))
    kwargs['ax'].xaxis.set_major_formatter(plt.FixedFormatter(names))

    plt.xticks(range(N)) # add loads of ticks
    plt.xticks(rotation='vertical')

    plt.gca().margins(x=0)
    plt.gcf().canvas.draw()
    tl = plt.gca().get_xticklabels()
    maxsize = max([t.get_window_extent().width for t in tl])
    m = 0.2 # inch margin
    s = maxsize/plt.gcf().dpi*N+2*m
    margin = m/plt.gcf().get_size_inches()[0]

    plt.gcf().subplots_adjust(left=margin, right=1.-margin)
    plt.gcf().set_size_inches(s, plt.gcf().get_size_inches()[1])

    # Tweak spacing to prevent clipping of tick-labels
    plt.subplots_adjust(bottom=-0.15)
    plt.savefig(join(kwargs['path'], kwargs['output']))
    print('File ', kwargs['output'], ' saved to ', kwargs['path'])
    plt.show()

'''
    group_reader(): Takes the period_community pairs and appends to dict if intersections occur. However,
    the returned dict requires furter analysis and processing, due to unknown order and
    content from the sorted and filtered communities, which is why they are then sent to
    the final_grouper by community_grouper, after completion here.
        - Args: Accepts the initial group dict, which is cross-referenced by the pair of
            period_community values extracted via a regex expression.
        - Returns: A dict of oversaturated comparisons, which are sent to final_grouper()
            for final analysis, reduction, and completion.
'''
def group_reader(group_dict, m1, m2):
    for g in list(group_dict): # parse list of dicts

        # If m1 in list, but not m2, append m2.
        if (m1 in group_dict[g]['matches']) and (m2 not in group_dict[g]['matches']):
            group_dict[g]['matches'].append(m2)
            return group_dict

        # Else if m1 not in list, but m2 is, append m1.
        elif (m1 not in group_dict[g]['matches']) and (m2 in group_dict[g]['matches']):
            group_dict[g]['matches'].append(m1)
            return group_dict

    # Get last key value from groups dict, and add 1 to it
    all_keys = group_dict.keys()
    last_key = list(all_keys)[-1]
    last_g = int(last_key) + 1

    # If union exists, rewrite list with union
    group_dict.update( { str(last_g): { 'matches': [m1, m2] } } )
    return group_dict

'''
    final_grouper(): Takes the period_community dictionaries and tests for their intersections.
        Then, it takes any intersections and joins them with .union and appends them into a
        localized running list, which will all be accrued in a running master list of that community.
        From there, each community result will be sorted by their length in descending order.
        - Args: Accepts the group dict from group_reader().
        - Returns: A dict of all unique period_community elements (2 or more) found to be similar.

'''
def final_grouper(**kwargs):
    inter_groups = []
    final_groups = []
    # Check for intersections first,
    # If intersection, check unions and assign
    # Append list of unions to inter_groups
    for a in kwargs['all_groups']:
        inter = {}
        inter_list = []
        union = {}
        for b in kwargs['all_groups']:
            inter = set(kwargs['all_groups'][a]['matches']).intersection( set(kwargs['all_groups'][b]['matches']) )
            if len(inter) >= 1:
                union = set(kwargs['all_groups'][a]['matches']).union( set(kwargs['all_groups'][b]['matches']) )
                inter_list.append(union)
        inter_groups.append(inter_list)

    # Sort list by their length in descending order
    descending_inter_groups = sorted(inter_groups, key=len)

    # Append first item as key
    for linter in descending_inter_groups:
        if list(linter)[0] not in final_groups:
            final_groups.append(list(linter)[0])
    return final_groups

'''
    community_grouper(): Controller function for process to group together communities found to be similar
    across periods in the corpus. It uses the 1) group_reader() and 2) final_grouper()
    functions to complete this categorization process.
        - Args: Accepts the network object (net_obj) with the returned value from nttc.match_maker(),
            which should be saved as .sorted_filtered_comms property: a list of tuples with
            sorted and filtered community pairs and their score, but it only uses the
            community values.
        - Returns: A list of sets, where each set is a grouped recurrent community:
            For example, 1_0, where 1 is the period, and 0 is the designated community
            number.

'''
def community_grouper(**kwargs):
    groups = {}
    communities_across_periods = []
    i = 0
    # These 2 patterns find the parts of the keys: 
    # Example: 10_6x3_20 becomes 10_6 and 3_20, respectively
    regex1 = r"(\b\w{1,2}_[^x]{1,2})"
    regex2 = r"(([^x]{1,5}\b))"
    if kwargs['top_mentions'] is True:
        for fcs in kwargs['match_obj'].sorted_filtered_mentions:
            # Parse comms into distinct strings for comparison
            match1 = re.findall(regex1, fcs[0])
            match2 = re.findall(regex2, fcs[0])
            communities_across_periods.append( (match1[0], match2[0][0]) )
            # If not values exist in groups, update
            if not groups:
                groups.update( {str(i): { 'matches': [ match1[0], match2[0][0] ] }} )
            # Else send matches to group_reader()
            else:
                group_reader(groups, match1[0], match2[0][0])
        fg = final_grouper(all_groups=groups)
        kwargs['match_obj'].groups_mentions = fg
        print('Be sure to double-check the output!')
        return kwargs['match_obj']
    elif kwargs['top_rters'] is True:
        for fcs in kwargs['match_obj'].sorted_filtered_rters:
            # Parse comms into distinct strings for comparison
            match1 = re.findall(regex1, fcs[0])
            match2 = re.findall(regex2, fcs[0])
            communities_across_periods.append( (match1[0], match2[0][0]) )
            # If not values exist in groups, update
            if not groups:
                groups.update( {str(i): { 'matches': [ match1[0], match2[0][0] ] }} )
            # Else send matches to group_reader()
            else:
                group_reader(groups, match1[0], match2[0][0])
        fg = final_grouper(all_groups=groups)
        kwargs['match_obj'].groups_rters = fg
        print('Be sure to double-check the output!')
        return kwargs['match_obj']
