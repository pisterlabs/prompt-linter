import guidedlda
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from sklearn.feature_extraction.text import CountVectorizer
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt
import numpy as np
from six.moves import cPickle as pickle

def build_model(data, num_topics, seed_topic_list, seed_conf, top_n = 10, include_vis = True):

    #form bow matrix to feed as input into training guidedlda model
    data = [' '.join(text) for text in data]
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(data).toarray()
    vocab = vectorizer.get_feature_names()
    word2id = dict((v, idx) for idx, v in enumerate(vocab))

    #Creates dictionary that assigns words to topics via their
    #topic id given by the id2word assignment
    seed_topics = {}
    for topic_id, subset in enumerate(seed_topic_list):
        for word in subset:
            if word in word2id:
                seed_topics[word2id[word]] = topic_id

    # Build GuidedLDA model
    guidedlda_model = guidedlda.GuidedLDA(n_topics = num_topics, n_iter = 100, random_state = 7, refresh = 20)
    guidedlda_model.fit(X, seed_topics = seed_topics, seed_confidence = seed_conf)

    top_vocab(guidedlda_model, vocab, top_n)

    # Saves model for production later
    with open('results/guided_lda/guided_lda_{}'.format(num_topics), 'wb') as f:
        pickle.dump(guidedlda_model, f)
    return guidedlda_model

    '''
    #pyLDAvis
    if include_vis:
        p = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
        pyLDAvis.save_html(p, 'lda_{}.html'.format(num_topics))
    '''

def get_best(data, seed_topic_list, seed_conf, lo, hi, step):
    tweets_coherence = []

    for nb_topics in range(lo, hi, step):
        guided_lda_model = build_model(data, nb_topics, seed_topic_list, seed_conf, include_vis = False)
        '''
        cohm = CoherenceModel(model=lda_model, corpus=corpus, dictionary=id2word, coherence='u_mass')
        coh = cohm.get_coherence()
        tweets_coherence.append(coh)

    # visualize coherence
    plt.figure(figsize=(10, 5))
    plt.plot(range(lo, hi, step), tweets_coherence)
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence Score");
    plt.savefig('lda_coherence.png', bbox_inches='tight')

    df = pd.DataFrame(data={"num_topics": range(lo, hi, step), "coherence": tweets_coherence})
    df.to_csv("coherence.csv", sep=',',index=False)
    '''

def top_vocab (guided_lda_model, vocab, num):
    top_words_per_topic = []

    topic_word = guided_lda_model.topic_word_
    for i, topic_dist in enumerate(topic_word):
        topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(num+1):-1].tolist()
        #top_words_per_topic.extend([(i,)+ for ])
        print('Topic {}: {}'.format(i, ' '.join(topic_words)))
    '''
    for t in range(lda_model.num_topics):
        top_words_per_topic.extend([(t,) + x for x in lda_model.show_topic(t, topn=num)])
    pd.DataFrame(top_words_per_topic, columns=['Topic', 'Word', 'P']).to_csv("top_words_{}.csv".format(lda_model.num_topics))
    '''

