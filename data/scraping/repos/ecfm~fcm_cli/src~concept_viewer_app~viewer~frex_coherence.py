import json
import os

import numpy as np
import scipy
import scipy.stats
import torch
from gensim.models.coherencemodel import CoherenceModel
from scipy.special import logsumexp

from toolbox.helper_functions import get_dataset

ROOT = os.path.abspath(__file__ + '/../../../..')
EPS = 1E-6


def load_dataset(grid_path, dataset, run_id):
    dataset_class = get_dataset(dataset)
    if dataset == 'csv':
        with open(os.path.join(grid_path, "config.json"), "r") as f:
            config = json.load(f)
        ds = dataset_class(config["csv-path"], config["csv-text"], config["csv-label"])
    else:
        ds = dataset_class()
    run_id_path = os.path.join(grid_path, run_id)
    with open(os.path.join(run_id_path, "params.json"), "r") as f:
        params = json.load(f)
    data = ds.load_data(params["dataset"])
    return data["vocab"], data["gensim_corpus"], data["gensim_dictionary"]


def load_dotproduct_distance(run_id_path, epoch):
    model_file = os.path.join(run_id_path, "model", "epoch%d.pytorch" % epoch)
    state = torch.load(model_file, map_location=torch.device('cpu'))
    topic_vectors = state['embedding_t'].cpu().numpy()
    word_vectors = state['embedding_i.weight'].cpu().numpy().T
    return -np.dot(topic_vectors, word_vectors)


def distance_to_distribution(distance, function=lambda x: abs(1 / (x + EPS)), normalize=True):
    """
    Input: T * V array. 
    """
    distribution = function(distance)
    if normalize:
        distribution = distribution / distribution.sum(axis=1, keepdims=True)
    return distribution


def ecdf(arr):
    """Calculate the empirical CDF values for all elements in a 1D array."""
    return scipy.stats.rankdata(arr, method='max') / arr.size


def calculate_frex(run_id_path, topics, w):
    """Calculate FREX for all words in a topic model.

    See R STM package for details:
    https://cran.r-project.org/web/packages/stm/vignettes/stmVignette.pdf

    """
    log_beta = np.log(topics)
    log_exclusivity = log_beta - logsumexp(log_beta, axis=0)
    exclusivity_ecdf = np.apply_along_axis(ecdf, 1, log_exclusivity)
    freq_ecdf = np.apply_along_axis(ecdf, 1, log_beta)
    np.save(os.path.join(run_id_path, "exclusivity_ecdf.npy"), exclusivity_ecdf)
    np.save(os.path.join(run_id_path, "freq_ecdf.npy"), freq_ecdf)
    return frex_score(exclusivity_ecdf, freq_ecdf, w)


def frex_score(exclusivity_ecdf, freq_ecdf, w):
    return 1. / (w / exclusivity_ecdf + (1 - w) / freq_ecdf)


def get_topics(vocab, frex, wordset, topn):
    topics = []
    for word_indices in (-frex).argsort(axis=1):
        words = []
        for i in word_indices:
            if vocab[i] in wordset:
                words.append(vocab[i])
            if len(words) >= topn:
                break
        topics.append(words)
    return topics


def get_coherence(topics, corpus, dic):
    coherence_model = CoherenceModel(topics=topics, corpus=corpus, dictionary=dic, coherence='u_mass')
    coherence_per_topic = '\n'.join(['%.2f' % c for c in coherence_model.get_coherence_per_topic()])
    return coherence_per_topic, coherence_model.get_coherence()


def get_top_frex_words(result, wordset, frex_w, topn):
    run_id_path = os.path.join(result.grid_path, result.run_id)
    vocab, corpus, dic = load_dataset(result.grid_path, result.dataset, result.run_id)
    excl_file = os.path.join(run_id_path, "exclusivity_ecdf.npy")
    freq_file = os.path.join(run_id_path, "freq_ecdf.npy")
    if os.path.exists(excl_file) and os.path.exists(freq_file):
        exclusivity_ecdf = np.load(excl_file)
        freq_ecdf = np.load(freq_file)
        frex = frex_score(exclusivity_ecdf, freq_ecdf, frex_w)
    else:
        distance = load_dotproduct_distance(run_id_path, result.epoch)
        if distance is None:
            return []
        distribution = distance_to_distribution(distance, normalize=False)
        frex = calculate_frex(run_id_path, distribution, w=frex_w)
    topics = get_topics(vocab, frex, wordset, topn)
    coherence_per_topic, coherence = get_coherence(topics, corpus, dic)
    return topics, coherence_per_topic, coherence
