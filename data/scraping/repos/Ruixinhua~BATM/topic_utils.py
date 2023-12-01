import heapq
import torch
import os
import numpy as np
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel

from utils.preprocess_utils import tokenize
from utils.general_utils import write_to_file


def get_topic_dist(trainer, word_seq):
    topic_dist = np.zeros((trainer.model.head_num, len(word_seq)))
    with torch.no_grad():
        bs = 512
        num = bs * (len(word_seq) // bs)
        word_feat = np.array(word_seq[:num]).reshape(-1, bs).tolist() + [word_seq[num:]]
        for words in word_feat:
            input_feat = {"data": torch.tensor(words).unsqueeze(0), "mask": torch.ones(len(words)).unsqueeze(0)}
            input_feat = trainer.load_batch_data(input_feat)
            _, topic_weight = trainer.best_model.extract_topic(input_feat)  # (B, H, N)
            topic_dist[:, words] = topic_weight.squeeze().cpu().data
        return topic_dist


def get_coherence(topics, texts, method):
    dictionary = Dictionary(texts)
    return CoherenceModel(topics=topics, texts=texts, dictionary=dictionary, coherence=method, topn=25)


def get_topic_list(matrix, top_n, reverse_dict):
    top_index = [heapq.nlargest(top_n, range(len(vec)), vec.take) for vec in matrix]
    topic_list = [[reverse_dict[i] for i in index] for index in top_index]
    return topic_list


def evaluate_topic(topic_list, data_loader):
    texts = [tokenize(s, data_loader.method) for s in data_loader.test_loader.dataset.texts]
    npmi = get_coherence(topic_list, texts, "c_npmi").get_coherence_per_topic()
    c_v = get_coherence(topic_list, texts, "c_v").get_coherence_per_topic()
    return npmi, c_v


def save_topic_info(path, weights, reverse_dict, data_loader, top_n=25):
    topic_list = get_topic_list(weights, top_n, reverse_dict)
    npmi, c_v = evaluate_topic(topic_list, data_loader)
    os.makedirs(path, exist_ok=True)
    write_to_file(os.path.join(path, "topic_list.txt"), [" ".join(topics) for topics in topic_list])
    topic_result = {"NPMI": np.mean(npmi), "CV": np.mean(c_v)}
    write_to_file(os.path.join(path, f"cv_coherence_{topic_result['CV']}.txt"), [str(s) for s in np.round(c_v, 4)])
    write_to_file(os.path.join(path, f"npmi_coherence_{topic_result['NPMI']}.txt"), [str(s) for s in np.round(npmi, 4)])
    return topic_result
