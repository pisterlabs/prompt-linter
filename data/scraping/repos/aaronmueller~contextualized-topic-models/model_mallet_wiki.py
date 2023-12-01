import os
import sys
import pickle
import numpy as np
# from littlebird import TweetReader, TweetTokenizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import gensim
from gensim import corpora, models, matutils
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.wrappers import LdaMallet
from gensim.models.wrappers.ldamallet import malletmodel2ldamodel

# NOTE: you will need to install Mallet for this script to work
MALLET_PATH = "/home/amueller/Mallet/bin/mallet"

def print_topics(model, count_vectorizer, n_top_words):
    words = count_vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(model.components_):
        print("\Topic #{}:".format(topic_idx))
        print(" ".join([words[i] for i in topic.argsort()[:-n_top_words-1:-1]]))


# read English data
docs = []; docs_test = []
texts = []; texts_test = []
with open("wiki_train_en_prep.txt", "r") as corpus:
    for line in corpus:
        docs.append(line.strip())
        texts.append(line.strip().split())

# data processing
num_topics = int(sys.argv[1])
num_words = 10
count_vectorizer = CountVectorizer(stop_words=stopwords.words('english'))
count_vectorizer.fit(docs)
doc_word = count_vectorizer.transform(docs).transpose()
corpus = matutils.Sparse2Corpus(doc_word)

# vocab creation
word2id = dict((v, k) for v, k in count_vectorizer.vocabulary_.items())
id2word = dict((v, k) for k, v in count_vectorizer.vocabulary_.items())
dictionary = corpora.Dictionary()
dictionary.id2token = id2word
dictionary.token2id = word2id

# topic modeling
ldamallet = LdaMallet(MALLET_PATH, corpus=corpus, num_topics=num_topics, id2word=id2word, iterations=400)

# save topic model to file
topic_file = open("english_topics_{}.pkl".format(sys.argv[1]), "wb")
pickle.dump(ldamallet.show_topics(formatted=False, num_topics=num_topics), topic_file)
topic_file.close()

# get NPMI coherence
coherence = CoherenceModel(model=ldamallet, texts=texts, dictionary=dictionary, coherence='c_npmi')
print("coherence:", coherence.get_coherence())
