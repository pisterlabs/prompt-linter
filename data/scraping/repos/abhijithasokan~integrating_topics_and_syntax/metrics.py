import os
from pprint import pprint
import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel
import numpy as np
import matplotlib.pyplot as plt

PHI_Z_FILE = "phi_z.txt"
PHI_C_FILE = "phi_c.txt"
dir_name = "output/0.3_0.3_0.1_0.2_10_5_5000_news_5100"

def plot(x,y,type):
    # Show graph
    plt.figure(figsize=(8,6))
    plt.plot(x, y,label=type)
    plt.title(f"Coherence score per iteration ({type})")
    plt.xlabel("n_iter")
    plt.ylabel("Coherence score")
    plt.legend()
    plt.xticks(rotation=45)

    plt.savefig(os.path.join(dir_name,f"{type}_n_iter_news.png"))

def get_data(type):
        if type=='test':
            dir_name = os.path.join(".","news_3000_test")
        else:
            dir_name = os.path.join(".","news_5100")
        with open(os.path.join(dir_name, "documents.txt"), "r") as documents_file:
            data = documents_file.read().splitlines()
            documents = [[int(w) for w in d.split(' ')] for d in data if d != '']
        with open(os.path.join(dir_name, "vocab.txt"), "r") as vocab_file:
            vocab = vocab_file.read().split(' ')
        return vocab,documents

def metrics(type,PHI_Z_FILE=PHI_Z_FILE, dir_name=dir_name):
    vocab,docs = get_data('train')
    vocab_test,docs_test = get_data('test')
    list_of_docs = []
    for doc in docs_test:
        doc_str_list = []
        for word_id in doc:
            word = vocab_test[word_id]
            doc_str_list.append(word)
        list_of_docs.append(doc_str_list)

    id2word = corpora.Dictionary(list_of_docs)
    corpus = [id2word.doc2bow(text) for text in list_of_docs]
    def get_top_k_words_from_topic(topic_id: int, k: int):
        top_k_words = np.argsort(topic_word_counts[topic_id])[:-k - 1:-1]
        return [vocab[word_id] for word_id in top_k_words]
    
    iter_list =[]
    coh_list = []
    topics=[]
    for subdir, _, files in os.walk(dir_name):
        for file in files:
            if 'iter' in subdir and PHI_Z_FILE==file:
                iter_name = subdir.split('\\')[1]
                topic_word_counts = np.loadtxt(os.path.join(subdir, PHI_Z_FILE))
                topics = [get_top_k_words_from_topic(t, 5) for t in range(10)]
                iter_list.append(iter_name)
                cm = CoherenceModel(topics=topics,texts = list_of_docs,corpus=corpus, dictionary=id2word, coherence=type)
                coh_list.append(cm.get_coherence())
    pprint(topics)
    plot(iter_list,coh_list,type)
    
def class_words():
    subdir = os.path.join(dir_name,f"iter_3600")
    vocab,docs = get_data('train')
    class_word_counts = np.loadtxt(os.path.join(subdir, PHI_C_FILE))
    def get_top_k_words_from_class( class_id: int, k: int):
            top_k_words = np.argsort(class_word_counts[class_id])[:-k - 1:-1]
            return [vocab[word_id] for word_id in top_k_words]
    return [get_top_k_words_from_class(t, 10) for t in range(5)]

if __name__ == "__main__":
    # metrics(type='c_v')
    pprint(class_words())