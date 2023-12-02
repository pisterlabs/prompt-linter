from sklearn.model_selection import StratifiedKFold          # for cross-evaluation
# from nltk.tokenize import RegexpTokenizer
# from nltk.stem.wordnet import WordNetLemmatizer

from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
from gensim.corpora.dictionary import Dictionary

import numpy as np

try:
    import cPickle as pickle    # since cPickle is faster than pickle, use that if possible
except:
    import pickle

import os
from pprint import pprint
from time import time

import matplotlib
matplotlib.use('agg')   # for linux env only (not required for windows
import matplotlib.pyplot as plt

from functions import plot_scores

os.chdir(r'/home/xzhangbx/remote/others/FYP/Topic-Modeling/source')     # depending on the path on server


learning_decay = 0.6
n_components = [11, 13, 15, 17]

max_iter = 200
valid_iter = 10
n_splits = 3

start_time = time()
dict_num_topic = {}
raw_text_dir = "../20newsgroup/processed_20newsgroup_all_news.txt"
model_dir = "./model/cross_validation_topicCoherence_n_topic_20 (learning decay"+str(learning_decay)+").pickle"

# (1) train the model if no score dictionary is found
if not os.path.isfile(model_dir):
    print "start loading dataset ..."

    file = open(raw_text_dir, 'r')
    texts = file.readlines()
    file.close()

    for i in range(len(texts)):
        texts[i] = texts[i].split()

    # dataset = pickle.load(open('../20newsgroup/processed_20newsgroup.pickle', 'r'))
    targets = pickle.load(open('../20newsgroup/20newsgroup_target.pickle', 'r'))

    # prepare dictionary and corpus
    dictionary = Dictionary(texts)
    print "shape of dataset: ", dictionary.num_docs, len(dictionary)                        # 18846 2567778 71127

    dictionary.filter_extremes(no_below=10, no_above=0.6)
    print "shape of dataset: ", dictionary.num_docs, dictionary.num_pos, len(dictionary)    # 18846 2567778 9198

    corpus = [dictionary.doc2bow(text) for text in texts]

    # TODO:store the placeholder into sparse matrix to save memory
    dataset_placeholder = np.empty(shape=(dictionary.num_docs, len(dictionary)))
    # passing to StratifiedKFold to obtain training and testing index

    for n_component in n_components:
        print '\n\n ------------------ # TOPIC =', n_component, ' -----------------------'
        lda_model = None
        cm = None

        skf = StratifiedKFold(n_splits=n_splits)
        splited_index = list(skf.split(X=dataset_placeholder, y=targets))   # skf.split returns a generator!

        train_scores = []        # size: (max_iter / valid_iter) * (n_splits)
        test_scores = []        # size: (max_iter / valid_iter) * (n_splits)
        train_perplexities = []  # size: (max_iter / valid_iter) * (n_splits)
        test_perplexities = []  # size: (max_iter / valid_iter) * (n_splits)

        for train_index, test_index in splited_index:
            train_s = []
            test_s = []
            train_p = []
            test_p = []

            print "test_index: ", type(test_index), ' ', test_index[:10]

            # here since corpus is a list, we cannot direclty use index list to slice
            train_data = [corpus[idx] for idx in train_index]
            test_data = [corpus[idx] for idx in test_index]

            for i in range(int(max_iter / valid_iter)):
                print '\ntraining ', i * valid_iter + 1, '-th iteration'

                if i == 0:
                    if lda_model != None:
                        del lda_model

                    lda_model = LdaModel(corpus=train_data, id2word=dictionary, num_topics=n_component, decay=learning_decay, iterations=valid_iter,
                                         random_state=0,)
                    # lda_model.update(train_data)
                else:
                    lda_model.update(corpus=train_data, decay=learning_decay, iterations=valid_iter)

                train_s.append(CoherenceModel(model=lda_model, corpus=train_data, dictionary=dictionary, coherence='u_mass').get_coherence())
                test_s.append(CoherenceModel(model=lda_model, corpus=test_data, dictionary=dictionary, coherence='u_mass').get_coherence())

                train_p.append(lda_model.log_perplexity(train_data))
                test_p.append(lda_model.log_perplexity(test_data))

            train_scores.append(train_s)
            test_scores.append(test_s)
            train_perplexities.append(train_p)
            test_perplexities.append(test_p)

            print "train_scores: ", train_scores[-1], " test_scores: ", test_scores[-1], \
                " train_perplexities: ", train_perplexities[-1], " test_perplexities: ", test_perplexities[-1]


        dict_num_topic[str(n_component) + '_topics'] = {
            "max_iter": max_iter, "valid_iter": valid_iter,
            "train_scores": train_scores, "test_scores": test_scores,
            "train_perplexities": train_perplexities, "test_perplexities": test_perplexities
        }

    pickle.dump(dict_num_topic, open(model_dir, 'w'))
    pprint(dict_num_topic)

else:
    dict_num_topic = pickle.load(open(model_dir, 'r'))

print "\nFinish Loading/Training within", time() - start_time, 'secends'
start_time = time()


print "start plotting..."

color_couples = [('#99c9eb', '#f998a5'), ('#4ca1dd', '#f77687'),
                 ('#0079cf', '#f6546a'),
                 ('#005490', '#c44354'), ('#003052', '#93323f'), ]
x_axis = range(0, max_iter, valid_iter)

plt.figure(figsize=(20, 16))
plt.suptitle('Tuning n_topic (learning decay = '+str(learning_decay)+')', fontsize=12)
ax_1 = plt.subplot(221)
ax_2 = plt.subplot(222)
ax_3 = plt.subplot(223)
ax_4 = plt.subplot(224)

for index, n_component in enumerate(n_components):
    (c1, c2) = color_couples[index]

    d = dict_num_topic[str(n_component) + '_topics']
    train_scores = np.array(d['train_scores'])
    test_scores = np.array(d['test_scores'])
    train_perplexities = np.array(d['train_perplexities'])
    test_perplexities = np.array(d['test_perplexities'])

    print " train_scores.shape: ", train_scores.shape
    print len(x_axis)


    alpha = 0.1
    label = 'topic_'+str(n_component)
    plot_range = False

    plot_scores(ax_1, x_axis, train_scores, n_splits, c1, title='train_scores', label=label, plot_range=plot_range)
    plot_scores(ax_2, x_axis, train_perplexities, n_splits, c1, title='train_perplexities', label=label, plot_range=plot_range)
    plot_scores(ax_3, x_axis, test_scores, n_splits, c2, title='test_scores', label=label, plot_range=plot_range)
    plot_scores(ax_4, x_axis, test_perplexities, n_splits, c2, title='test_perplexities', label=label, plot_range=plot_range)

    plt.savefig('./GENSIM_numT_decay'+str(learning_decay)+'_noRange.png')

    print "\nFinish Plotting within", time() - start_time, 'secends'
