import heapq
import os

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
from scipy.stats import entropy
from typing import Dict, Union, List
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from modules.utils import word_tokenize, get_project_root


class NPMI:
    """
    Reference: https://github.com/ahoho/gensim-runner/blob/main/utils.py
    NPMI (Normalized Pointwise Mutual Information) is a measure of association between two words: usually used to 
    evaluate topic quality.
    """

    def __init__(
            self,
            bin_ref_counts: Union[np.ndarray, sparse.spmatrix],
            vocab: Dict[str, int] = None,
    ):
        assert bin_ref_counts.max() == 1
        self.bin_ref_counts = bin_ref_counts
        if sparse.issparse(self.bin_ref_counts):
            self.bin_ref_counts = self.bin_ref_counts.tocsc()
        self.npmi_cache = {}  # calculating NPMI is somewhat expensive, so we cache results
        self.vocab = vocab

    def compute_npmi(
            self,
            beta: np.ndarray = None,
            topics: Union[np.ndarray, List] = None,
            vocab: Dict[str, int] = None,
            n: int = 10
    ) -> np.ndarray:
        """
        Compute NPMI for an estimated beta (topic-word distribution) parameter using
        binary co-occurence counts from a reference corpus

        Supply `vocab` if the topics contain terms that first need to be mapped to indices
        """
        if beta is not None and topics is not None:
            raise ValueError(
                "Supply one of either `beta` (topic-word distribution array) "
                "or `topics`, a list of index or word lists"
            )
        if vocab is None and any([isinstance(idx, str) for idx in topics[0][:n]]):
            raise ValueError(
                "If `topics` contains terms, not indices, you must supply a `vocab`"
            )

        if beta is not None:
            topics = np.flip(beta.argsort(-1), -1)[:, :n]
        if topics is not None:
            topics = [topic[:n] for topic in topics]
        if vocab is not None:
            assert (len(vocab) == self.bin_ref_counts.shape[1])
            topics = [[vocab[w] for w in topic[:n]] for topic in topics]

        num_docs = self.bin_ref_counts.shape[0]
        npmi_means = []
        for indices in topics:
            npmi_vals = []
            for i, idx_i in enumerate(indices):
                for idx_j in indices[i + 1:]:
                    ij = frozenset([idx_i, idx_j])
                    try:
                        npmi = self.npmi_cache[ij]
                    except KeyError:
                        col_i = self.bin_ref_counts[:, idx_i]
                        col_j = self.bin_ref_counts[:, idx_j]
                        c_i = col_i.sum()
                        c_j = col_j.sum()
                        if sparse.issparse(self.bin_ref_counts):
                            c_ij = col_i.multiply(col_j).sum()
                        else:
                            c_ij = (col_i * col_j).sum()
                        if c_ij == 0:
                            npmi = 0.0
                        else:
                            npmi = (
                                    (np.log(num_docs) + np.log(c_ij) - np.log(c_i) - np.log(c_j))
                                    / (np.log(num_docs) - np.log(c_ij))
                            )
                        self.npmi_cache[ij] = npmi
                    npmi_vals.append(npmi)
            npmi_means.append(np.mean(npmi_vals))

        return np.array(npmi_means)


def load_sparse(input_file):
    return sparse.load_npz(input_file).tocsr()


def extract_topics_base(model, word_seq, device):
    voc_size = len(word_seq)
    model = model.to(device)
    try:
        # the number of heads is the number of topics
        topic_dist = np.zeros((model.head_num, voc_size))
    except AttributeError:
        model = model.module  # for multi-GPU
        topic_dist = np.zeros((model.head_num, voc_size))
    with torch.no_grad():
        word_feat = {"news": torch.tensor(word_seq).unsqueeze(0).to(device), "evaluate_topic": True,
                     "news_mask": torch.ones(len(word_seq)).unsqueeze(0).to(device)}
        topic_dict = model.extract_topic(word_feat)  # (B, H, N)
        topic_weight = topic_dict["topic_weight"]
        topic_dist[:, word_seq] = topic_weight.squeeze().cpu().data
    return topic_dist


def extract_topics(model: torch.nn.Module, word_dict, device):
    word_seq = list(word_dict.values())
    topic_dist = extract_topics_base(model, word_seq, device)  # global topics (base)
    return topic_dist


def get_topic_dist(model, word_dict):
    """
    Get topic distribution matrix
    :param model: model with extract_topic method
    :param word_dict: target word dictionary
    :return: An numpy array with shape (topic_num, word_num) of the topic distribution matrix
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # only run on one GPU
    try:
        topic_dist = extract_topics(model, word_dict, device)
    except (RuntimeError,):  # RuntimeError: CUDA out of memory, change to CPU
        device = torch.device("cpu")
        topic_dist = extract_topics(model, word_dict, device)
    return topic_dist


def get_topic_list(matrix, top_n, reverse_dict):
    """input topic distribution matrix is made up of (topic, word)"""
    top_index = [heapq.nlargest(top_n, range(len(vec)), vec.take) for vec in matrix]
    topic_list = [[reverse_dict[i] for i in index] for index in top_index]
    return topic_list


def compute_coherence(topic_list, texts, **kwargs):
    dictionary = Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    cm = CoherenceModel(topics=topic_list, texts=texts, dictionary=dictionary, corpus=corpus, **kwargs)
    return cm.get_coherence_per_topic()


def evaluate_entropy(topic_dist):
    token_entropy, topic_entropy = np.mean(entropy(topic_dist, axis=0)), np.mean(entropy(topic_dist, axis=1))
    return token_entropy, topic_entropy


def cal_topic_diversity(topic_words):
    """topic_words is in the form of [[w11,w12,...],[w21,w22,...]]"""
    vocab = set(sum(topic_words, []))
    n_total = len(topic_words) * len(topic_words[0])
    topic_div = len(vocab) / n_total
    return topic_div


def slow_topic_eval(config, topic_list):
    """
    Slow topic evaluation using gensim
    :param config: Configuration object
    :param topic_list: list of topics, each topic is a list of terms
    :return: coherence score
    """
    tokenized_method = config.get("tokenized_method", "keep_all")
    ws = config.get("window_size", 200)
    ps = config.get("processes", 35)
    tokenized_data_path = Path(get_project_root()) / f"dataset/data/MIND_tokenized.csv"
    ref_df = pd.read_csv(config.get("slow_ref_data_path", tokenized_data_path))
    ref_texts = [word_tokenize(doc, tokenized_method) for doc in ref_df["tokenized_text"].values]
    topic_score = {
        m: compute_coherence(
            topic_list, ref_texts, coherence=m, topn=config.get("top_n", 10), window_size=ws, processes=ps
        ) for m in config.get("coherence_method", ["c_npmi"])
    }
    return topic_score


def fast_npmi_eval(config, topic_list, word_dict):
    """
    Fast NPMI evaluation using pre-computed sparse matrix
    :param config: Configuration object
    :param topic_list: list of topics, each topic is a list of words
    :param word_dict: dictionary of topic words
    :return: npmi score
    """
    ref_data_path = config.get("ref_data_path")
    if ref_data_path is None or not os.path.exists(ref_data_path):
        raise ValueError("ref_data_path is not specified")
    ref_texts = load_sparse(ref_data_path)
    scorer = NPMI((ref_texts > 0).astype(int))
    topic_index = [[word_dict[word] - 1 for word in topic] for topic in topic_list]
    # convert to index list: minus 1 because the index starts from 0 (0 is for padding)
    return scorer.compute_npmi(topics=topic_index, n=config.get("top_n", 10))


def w2v_sim_eval(config, embeddings, topic_list, word_dict):
    top_n = config.get("top_n", 10)
    count = top_n * (top_n - 1) / 2
    topic_index = [[word_dict[word] for word in topic] for topic in topic_list]
    return [np.sum(np.triu(cosine_similarity(embeddings[i]), 1)) / count for i in topic_index]
