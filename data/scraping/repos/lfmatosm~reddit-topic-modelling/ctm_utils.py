from gensim.models import CoherenceModel
import functools
from collections import Counter
from decimal import Decimal
from gensim.corpora import Dictionary
from sklearn.feature_extraction.text import CountVectorizer
from scipy.spatial import distance
from itertools import combinations
from pymongo import MongoClient

client = MongoClient('localhost', 27017)
db = client.word_embeddings_300d
collection = db.embeddings


def create_dictionary(documents):
    """Creates word dictionary for given corpus.

    Parameters:
    
    documents (list of str): set of documents

    Returns:

    dictionary (gensim.corpora.Dictionary): gensim dicionary of words from dataset
    """
    dictionary = Dictionary(documents)
    dictionary.compactify()

    return dictionary


def filter_tokens_by_frequencies(documents, min_df=1, max_df=1.0):
    vectorizer = CountVectorizer(min_df=min_df, max_df=max_df)
    vectorizer.fit_transform(documents)
    
    return [[word for word in document if word not in vectorizer.stop_words_] for document in documents]


def get_words_from_db(word1, word2):
    print(f'word1={word1}, word2={word2}')
    w1 = collection.find_one({ "word": word1 })
    w2 = collection.find_one({ "word": word2 })

    print(f'w1={w1}, w1={w2}')

    def f(x): return float(x)

    return list(map(f, w1)), list(map(f, w2))


def pairwise_word_embedding_distance(topics, topk=20):
    """
    :param topk: how many most likely words to consider in the evaluation
    :return: topic coherence computed on the word embeddings similarities
    """
    if topk > len(topics[0]):
        raise Exception('Words in topics are less than topk')
    else:
        sum_dist = 0
        for topic in topics:
            dist = 0
            combs = combinations(topic[:topk], 2)
            for word1, word2 in combs:
                w1, w2 = get_words_from_db(word1, word2)
                dist += distance.cosine(w1, w2)
            sum_dist += dist / topk
        return sum_dist / len(topics)
