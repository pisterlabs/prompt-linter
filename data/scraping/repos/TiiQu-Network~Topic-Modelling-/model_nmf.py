import numpy as np
import matplotlib.pyplot as plt
import wordcloud
from gensim.models.nmf import Nmf
from gensim.models import CoherenceModel
from sklearn.decomposition import NMF
from operator import itemgetter
''' This module aims to implement Non-Negative Matrix Factorization as a technique for topic modeling and since this approach has been proved to be efficient
in terms of time complexity, it can be employed to have an estimation of topic numbers.
'''
def NMF(corpus, num_topics, dic):
    nmf = Nmf(corpus=corpus, num_topics=num_topics, id2word=dic,chunksize=2000,
        passes=5,
        kappa=.1,
        minimum_probability=0.01,
        w_max_iter=300,
        w_stop_condition=0.0001,
        h_max_iter=100,
        h_stop_condition=0.001,
        eval_every=10,
        normalize=True,
        random_state=42)
    return nmf

def get_coherence(model, word_bigrams, dic):
    cm = CoherenceModel(
        model=model,
        texts=word_bigrams,
        dictionary=dic,
        coherence='c_v')
    return round(cm.get_coherence(), 5)

def num_topics_estimation(corpus, dic, word_bigrams, lower_bound, upper_bound, step=1):
    topic_nums = list(np.arange(lower_bound, upper_bound, step))
    coherence_scores = []
    for num in topic_nums:
        nmf = NMF(corpus, num, dic)
        coherence_scores.append(get_coherence(nmf, word_bigrams, dic))
    scores = list(zip(topic_nums, coherence_scores))
    best_num_topics = sorted(scores, key=itemgetter(1), reverse=True)[0][0]
    print('%d topics would be estimated for this data' % best_num_topics)
    return best_num_topics, scores

def ilustrate_word_clouds(model):
    for t in range(model.num_topics):
        plt.figure()
        plt.imshow(wordcloud.WordCloud().fit_words(dict(model.show_topic(t, 10))))
        plt.axis("off")
        plt.title("Topic #" + str(t))
        plt.show()
    
