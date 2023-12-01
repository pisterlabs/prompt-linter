# -*- coding: utf-8 -*-
# -*- train.py -*-


import os 
import time
import faiss
import pickle
import gensim
import string
import argparse
import unicodedata

import numpy as np
import pandas as pd
import multiprocessing as mp

from collections import Counter
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from gensim import models, corpora
from gensim.models import LdaModel, nmf
from gensim.models.coherencemodel import CoherenceModel
from sklearn import metrics
from sklearn.mixture import GaussianMixture
from sklearn.feature_extraction.text import TfidfVectorizer


def tokenize(s):
    s = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
    s = ''.join(c for c in s if (c not in string.punctuation) or (c in (' ', '-', '_'))).lower()
    t = RegexpTokenizer(r'\w+').tokenize(s)
    t = [w.replace('-', '_').lower() for w in t if w not in stop_words and w != '' and w != '-']
    return t


def word_embedding_coherence(topics, glove_embeddings):
    c_we_per_topic = []
    c_we_std_per_topic = []

    for topic in topics:
        scores = []
        for i in range(len(topic)):
            for j in range(i+1, len(topic)):
                word1 = topic[i]
                word2 = topic[j]

                if word1 not in glove_embeddings or word2 not in glove_embeddings: 
                    print(word1, 'or', word2, 'not in GloVe')
                    continue

                scores.append(glove_embeddings.similarity(word1, word2))

        c_we_per_topic.append(0 if len(scores) == 0 else np.mean(scores))
        c_we_std_per_topic.append(0 if len(scores) == 0 else np.std(scores))

    c_we = np.mean(c_we_per_topic)
    c_we_std = np.std(c_we_per_topic)
    
    return c_we, c_we_std, c_we_per_topic, c_we_std_per_topic

def get_related_words(word, thresh, vocabulary):
    try:
        neighborhood = pickle.load(open('/data/cache/neighborhoods/'+word+'.pickle', 'rb'))
    except:
        neighborhood = {word:{'sim': 1.0}}
    return word, {ww:neighborhood[ww]['sim'] for ww in neighborhood if ww in vocabulary and neighborhood[ww]['sim'] > thresh}

def get_related_vocabulary(word):
    return get_related_words(word, thresh=0, vocabulary=vocabulary)

parser = argparse.ArgumentParser(description='Topic Model Training')
parser.add_argument("-m", "--model", type=str, help="Which model to user", choices=['lda', 'kmeans', 'gmm', 'nmf'], default='kmeans')
parser.add_argument("-n", "--number_of_topics", type=str, help="Number of topics ('auto' = number of labels)", default='auto')
parser.add_argument("-t", "--thresh", type=str, help="Cut-off threshold for numberbatch similarity", default=0.15)
parser.add_argument("-c", "--classes", type=str, default='all', help="Specify which labels to keep (comma-separated, or 'all')")
parser.add_argument("-d", "--dataset_path", type=str, help="Path to the dataset CSV", default='data/bbc.csv', required=True)
parser.add_argument("-i", "--iterations", type=int, help="Maximum number of iterations", default=10)
parser.add_argument("-p", "--preprocessing", type=int, help="Specify which preprocessing pipeline to use", default=0)
parser.add_argument("-r", "--redo", type=int, help="Number of runs (with different seeds) to do", default=1)
parser.add_argument("-s", "--save_path", type=str, help="Path to save the results", default='results/')
parser.add_argument("-e", "--cache_path", type=str, help="Where to cache preprocessed data", default='/data/cstm_cache/')
parser.add_argument("-x", "--external_vocab", help="Whether or not to add external vocabulary", action='store_true', default=True)
parser.add_argument("-g", "--glove_path", type=str, help="Path to pickled GloVe embeddings", default='../../Topic-Model-API/tomodapi/glove/glove.6B.300d.pickle')

args = vars(parser.parse_args())

dataset_name = args['dataset_path'].split('/')[-1].split('.')[0]
cache_path = args['cache_path']
n_topics = len(dataset.label.unique()) if args['number_of_topics'] == 'auto' else int(args['number_of_topics'])
experiment_id = '_'.join([str(int(time.time())), args['model'], dataset_name, str(n_topics), str(args['iterations'])])

print('Loading GloVe embeddings..')
glove_embeddings = pickle.load(open(args['glove_path'], 'rb'))

stop_words = list(stopwords.words('english')) 
stop_words.extend(['could', "he'd", "he'll", "he's", "here's", "how's", "i'd", "i'll", "i'm", "i've", "let's",
                   'ought', "she'd", "she'll", "that's", "there's", "they'd", "they'll", "they're", "they've",
                   "we'd", "we'll", "we're", "we've", "what's", "when's", "where's", "who's", "why's", 'would',
                   'also'])

dataset = pd.read_csv(args['dataset_path'])

try:
    assert('text' in dataset.columns and 'label' in dataset.columns)
except:
    raise Exception('Either "text" or "label" columns are missing in the provided dataset')
    
if args['classes'] != 'all':
    classes_to_keep = args['classes'].split(',')
    if any([cls not in dataset.label.unique() for cls in classes_to_keep]):
        raise Exception('Some of the specified classes are not in provided dataset')
    
    dataset = dataset[dataset.label.isin(classes_to_keep)]

print('Dataset', dataset_name, 'loaded, it contains', len(dataset), 'documents.')


with mp.Pool(mp.cpu_count()) as pool:
    texts_raw, gt_labels = pool.map(tokenize, dataset.text), list(dataset.label)
    
print('Dataset preprocessed. Labels distribution:')
print(Counter(gt_labels))

dictionary = corpora.Dictionary(texts_raw)
vocabulary = sorted(dictionary.token2id.keys())

    
"""
for value in [0.05, 0.1, 0.15, 0.2, 0.5]:
    args['thresh'] = value
"""


if args['model'] in ['kmeans', 'gmm']:
    if os.path.exists(cache_path+dataset_name+'_related_vocab.pickle'):
        print('Loading related vocabulary from cache.')
        related_vocab = pickle.load(open(cache_path+dataset_name+'_related_vocab.pickle', 'rb'))
    else:
        print('Caching related vocabulary..')
        with mp.Pool(mp.cpu_count()) as pool:
            words_and_related_vocab = pool.map(get_related_vocabulary, vocabulary)
            related_vocab = dict(words_and_related_vocab)
        pickle.dump(related_vocab, open(cache_path+dataset_name+'_related_vocab.pickle', 'wb'))
        print('Related vocabulary saved into cache.')

    related_vocab_path = cache_path+dataset_name+'_related_vocab_thresh_'+str(args['thresh'])+'.pickle'
    print("processing related vocabulary for threshold", args['thresh'])

    if os.path.exists(related_vocab_path):
        print('Loading related thresh vocabulary from cache.. ', end='')
        related_vocab = pickle.load(open(related_vocab_path, 'rb'))
        print('loaded!')
    else:
        print('No vocabulary cache found at ', related_vocab_path)
        related_vocab_thresh = {}
        for word in tqdm(related_vocab):
            related_vocab_thresh[word] = {w:related_vocab[word][w] for w in related_vocab[word] 
                                          if w in vocabulary and related_vocab[word][w] > args['thresh']}
        pickle.dump(related_vocab_thresh, open(related_vocab_path, 'wb'))
        print('Related thresh ('+str(args['thresh'])+') vocabulary saved into cache.')

    related_vocab = pickle.load(open(related_vocab_path, 'rb'))

    
os.system('mkdir results/'+experiment_id)
os.chmod('results/'+experiment_id, 0o777)
for random_state in range(args['redo']):
    if args['model'] == 'lda':
        corpus_bow = [dictionary.doc2bow(text) for text in texts_raw]
        
        time_to_train = time.time()
        lda = models.LdaModel(corpus=corpus_bow, 
                               id2word=dictionary, 
                               num_topics=n_topics, 
                               passes=args['iterations'], 
                               random_state=random_state)
        os.system('mkdir results/'+experiment_id+'/model_'+str(random_state)+'/')
        lda.save('results/'+experiment_id+'/model_'+str(random_state)+'/model')
        time_to_train = time.time() - time_to_train
        
        topic_preds = [max(lda[doc], key=lambda x: x[1])[0] for doc in corpus_bow]
        time_to_preds = time.time() - time_to_train
        
        topic_words = [[tw[0] for tw in lda.show_topic(r)] for r in range(n_topics)]
        
    if args['model'] == 'nmf':
        corpus_bow = [dictionary.doc2bow(text) for text in texts_raw]
        
        time_to_train = time.time()
        nmf = models.nmf.Nmf(corpus=corpus_bow, 
                                id2word=dictionary, 
                                num_topics=n_topics, 
                                random_state=random_state)
        
        os.system('mkdir results/'+experiment_id+'/model_'+str(random_state)+'/')
        lda.save('results/'+experiment_id+'/model_'+str(random_state)+'/model')
        time_to_train = time.time() - time_to_train
        
        topic_preds = [max(lda[doc], key=lambda x: x[1])[0] for doc in corpus_bow]
        time_to_preds = time.time() - time_to_train
        
        topic_words = [[tw[0] for tw in lda.show_topic(r)] for r in range(n_topics)]
        
    elif args['model'] == 'kmeans':
        corpus_cs = []

        for d in texts_raw:
            original_doc_words = d
            final_doc = []
            for word in original_doc_words:
                final_doc.extend([word] if word not in related_vocab else list(related_vocab[word].keys()))
            corpus_cs.append(final_doc)
            
        time_to_train = time.time()
        vectorizer = TfidfVectorizer(min_df=2, stop_words='english')
        X = vectorizer.fit_transform([' '.join(d) for d in corpus_cs]).toarray()
        id2word = {w:i for i,w in vectorizer.vocabulary_.items()}
        
        kmeans = faiss.Kmeans(d=X.shape[1], k=n_topics, niter=args['iterations'], nredo=1, seed=random_state)
        kmeans.train(X.astype(np.float32))
        time_to_train = time.time() - time_to_train

        topic_preds = kmeans.index.search(X.astype(np.float32), 1)[1].reshape((X.shape[0], ))
        time_to_preds = time.time() - time_to_train

        cluster_centers = kmeans.centroids
        topic_words = []
        for cluster in cluster_centers:
            sorted_values = sorted(list(enumerate(cluster)), key=lambda x: -x[1])
            top_words = [(id2word[w[0]], w[1]) for w in sorted_values[:10]]
            topic_words.append([w[0] for w in top_words])

    elif args['model'] == 'gmm':
        corpus_cs = []

        for d in texts_raw:
            original_doc_words = d
            final_doc = []
            for word in original_doc_words:
                if word not in related_vocab:
                    final_doc.extend([word])
                else:
                    final_doc.extend(list([w for w in related_vocab[word][word].keys() if len(w) > 2]))
            corpus_cs.append(final_doc)
        
        vectorizer = TfidfVectorizer(min_df=2, stop_words='english')
        X = vectorizer.fit_transform([' '.join(d) for d in corpus_cs]).toarray()
        id2word = {w:i for i,w in vectorizer.vocabulary_.items()}
        
        gmm = GaussianMixture(n_components=n_topics, max_iter=2, random_state=random_state).fit(X[:10])
        topic_preds = gmm.predict(X)
        cluster_centers = gmm.means_
        topic_words = []
        for cluster in cluster_centers:
            sorted_values = sorted(list(enumerate(cluster)), key=lambda x: -x[1])
            top_words = [(id2word[w[0]], w[1]) for w in sorted_values[:10]]
            print(top_words)
            topic_words.append([w[0] for w in top_words])
            
    completeness = metrics.completeness_score(gt_labels, topic_preds)
    homogeneity  = metrics.homogeneity_score(gt_labels, topic_preds)
    v_measure    = metrics.v_measure_score(gt_labels, topic_preds)

    c_we, c_we_std, c_we_per_topic, c_we_std_per_topic = word_embedding_coherence(topic_words, glove_embeddings)

    cm = CoherenceModel(topics=topic_words, texts=texts_raw, dictionary=dictionary, coherence='c_npmi')
    coherence = cm.get_coherence() 
    time_to_eval = time.time() - time_to_preds


    run = {}
    run['args'] = args
    run['topics'] = topic_words
    run['times'] = {'train': time_to_train, 'pred': time_to_preds, 'eval':time_to_eval}
    run['metrics'] = {'completeness': completeness, 
                      'homogeneity': homogeneity, 
                      'v_measure':v_measure,
                      'coherence':coherence,
                      'wb_coherence': (c_we, c_we_std, c_we_per_topic, c_we_std_per_topic)}

    pickle.dump(run, open(args['save_path'] + experiment_id + '/results.pickle', 'wb'))
    
    print('RUN', random_state)
    print(run)