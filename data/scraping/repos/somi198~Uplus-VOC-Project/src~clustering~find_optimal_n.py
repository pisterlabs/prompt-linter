# -*- coding: utf-8 -*-
from gensim.models.ldamodel import LdaModel
import logging
import pickle
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt
import argparse

class Documents:
    def __init__(self, path):
        self.path = path

    def __iter__(self):
        with open(self.path, encoding='utf-8') as f:
            for doc in f:
                yield doc.strip().split()


class Corpus:
    def __init__(self, path, dictionary):
        self.path = path
        self.dictionary = dictionary
        self.length = 0

    def __iter__(self):
        with open(self.path, encoding='utf-8') as f:
            for doc in f:
                yield self.dictionary.doc2bow(doc.split())

    def __len__(self):
        if self.length == 0:
            with open(self.path, encoding='utf-8') as f:
                for i, doc in enumerate(f):
                    continue
            self.length = i + 1
        return self.length

def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """ Compute c_v coherence for various number of topics
    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values


def find_optimal_number_of_topics(dictionary, corpus, processed_data):
    limit = 40;
    start = 2;
    step = 6;

    model_list, coherence_values = compute_coherence_values(dictionary=dictionary, corpus=corpus, texts=processed_data,
                                                            start=start, limit=limit, step=step)

    x = range(start, limit, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    plt.show()


parser = argparse.ArgumentParser(description='set option')
parser.add_argument("--file_name", type=str, default='service_center',
                    help="please enter file name   ex) service_center")
args = parser.parse_args()
fname = args.file_name


with open('/content/drive/Shareddrives/capstone/Elegant_Friends/rsc/clustering_data/'+fname+'/pickle/corpus.pickle', 'rb') as f:
    corpus = pickle.load(f)
with open('/content/drive/Shareddrives/capstone/Elegant_Friends/rsc/clustering_data/'+fname+'/pickle/dictionary.pickle', 'rb') as f2:
    dictionary = pickle.load(f2)
with open('/content/drive/Shareddrives/capstone/Elegant_Friends/rsc/clustering_data/'+fname+'/pickle/documents.pickle', 'rb') as f3:
    documents = pickle.load(f3)
    

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# 최적의 토픽 수 찾기
find_optimal_number_of_topics(dictionary, corpus, documents)
