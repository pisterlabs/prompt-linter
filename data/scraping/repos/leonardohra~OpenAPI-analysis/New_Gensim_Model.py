import numpy as np
import pandas as pd
import csv

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
from gensim.models.lsimodel import LsiModel
from gensim.models.nmf import Nmf
from gensim import similarities

from gensim.corpora import MmCorpus
from gensim.test.utils import get_tmpfile

from nltk.corpus import stopwords
from CustomEnumerators import TopicModelingAlgorithm, CoherenceType

class Gensim_Model():

    __data_lemmatized = None
    __id2words = None
    __corpus = None
    __current_model = None
    __best_model_par = None

    def __init__(self, data_lemmatized, id2words_file = None, corpus_file = None):
        self.__data_lemmatized = data_lemmatized

        self.__id2words = corpora.Dictionary.load(id2words_file) if id2words_file != None else corpora.Dictionary(data_lemmatized)
        self.__id2words.save('id2words.dict')

        self.__corpus = MmCorpus(corpus_file) if corpus_file != None else [self.__id2words.doc2bow(text) for text in data_lemmatized]
        MmCorpus.serialize('BoW_corpus.mm', self.__corpus)


    def __create_model(self, algo, topic_qtt):
        model = None
        if(algo == TopicModelingAlgorithm.LDA):
            model = gensim.models.ldamodel.LdaModel(corpus=self.__corpus, num_topics=topic_qtt, id2word=self.__id2words, random_state=1, per_word_topics=True)
        elif(algo == TopicModelingAlgorithm.LSA):
            model = gensim.models.lsimodel.LsiModel(corpus=self.__corpus, num_topics=topic_qtt, id2word=self.__id2words)
        elif(algo == TopicModelingAlgorithm.NMF):
            model = gensim.models.nmf.Nmf(corpus=self.__corpus, num_topics=topic_qtt, id2word=self.__id2words, random_state=1)

        return model

    def __create_lda(self, topic_qtt):
        lda_model = gensim.models.ldamodel.LdaModel(corpus=self.__corpus,
                                           id2word=self.__id2word,
                                           num_topics=topic_qtt,
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)
        return lda_model

    def __compute_coherence(self, model, tn):
        c_v = CoherenceModel(model=model, texts=self.__data_lemmatized, corpus=self.__corpus, dictionary=self.__id2words, coherence='c_v', topn=tn)
        return c_v.get_coherence()

    def set_model(self, algo, topic_qtt):
        self.__current_model = self.__create_model(algo, topic_qtt)

    def current_coherence(self, topn=20):
        if(self.__current_model != None):
            return self.__compute_coherence(self.__current_model, topn)
        else:
            return None

    def evaluate_several_models(self, algorithms = [TopicModelingAlgorithm.LDA, TopicModelingAlgorithm.LSA, TopicModelingAlgorithm.NMF], qtt_topics = range(2,21), topns = [5], out_folder=None, out_file=None):
        all_values = {
            CoherenceType.U_MASS: [],
            CoherenceType.C_V: [],
            CoherenceType.C_UCI: [],
            CoherenceType.C_NPMI: []
        }
        best_par = {
            CoherenceType.U_MASS: [None, None, -99999],
            CoherenceType.C_V: [None, None, -99999],
            CoherenceType.C_UCI: [None, None, -99999],
            CoherenceType.C_NPMI: [None, None, -99999]
        }

        topics_all = []

        # for num_topics_value in num_topics_list:
        for qtt in qtt_topics:
            for algorithm in algorithms:
                for topn in topns:
                    model = self.__create_model(algorithm, qtt)
                    topics = model.show_topics(num_topics=qtt ,formatted=False, num_words=topn)
                    c = self.__compute_coherence(model, topn)

                    all_values[CoherenceType.C_V].append([algorithm, qtt, topn, c])

                    print('{0}, {1}, {2}, {3}, {4}'.format(algorithm, 'c_v', qtt, topn, c))

                    for topic in topics:
                        algo = algorithm
                        qtt = qtt
                        topic_n = topic[0]
                        words_join = []
                        top_words = topn

                        if algo == TopicModelingAlgorithm.NMF:
                            words = ['{:.2f} * {}'.format(round(word[1], 2), word[0]) for word in topic[1]]
                            words_join = ' + '.join(words)
                        else:
                            words = ['{:.2f} * {}'.format(round(word[1], 2), word[0]) for word in topic[1]]
                            words_join = ' + '.join(words)

                        topics_all.append([algorithm.simple_name(), qtt, top_words, topic_n, words_join])

                    #print('Done for {} topics'.format(qtt))

                    if(c > best_par[CoherenceType.C_V][-1]):
                        #print('\n\n\nNew best parameters! \nAlgorithm: {0}\nCoherence Type: {1}\nCoherence value: {2} \nNumber of Topics: {3}\n'.format(str(algorithm), str(coherence), c, qtt))
                        best_par[CoherenceType.C_V] = [algorithm, qtt, c]


        self.__best_model_par = best_par[CoherenceType.C_V]
        if(out_folder != None):
            w = csv.writer(open(out_folder + out_file + '.csv', 'w'))
            w2 = csv.writer(open(out_folder + out_file + '_topics.csv', 'w'))


            for values in all_values[CoherenceType.C_V]:
                w.writerow(values)

            for topics in topics_all:
                w2.writerow(topics)

    def set_best_model(self):
        if(self.__best_model_par != None):
            self.__current_model = self.__create_model(self.__best_model_par[0], self.__best_model_par[1])
            return True

        return False

    def get_topic_distribution(self, sentence_tk):
        res = []
        qtt = self.__current_model.num_topics
        vect = self.__id2words.doc2bow(sentence_tk)
        dist = self.__current_model[vect]

        for i in range(qtt):
            found = False
            for j in range(len(dist)):
                if(i == dist[j][0]):
                    res.append([i, dist[j][1]])
                    found = True
                    break

            if(not found):
                res.append([i, 0])

        return res

    def new_text_to_topics(self, sentence_tk):
        id_to_tpc = []
        dist_p_tpc = self.get_topic_distribution(sentence_tk)
        for tpc in dist_p_tpc:
            id_to_tpc.append([123456789, tpc[0], tpc[1]])

        df2 = pd.DataFrame(id_to_tpc, columns=['id_sentence', 'topic_num', 'score'])

        return df2

    def endpoint_in_topics_dataset(self):
        endp_tpcs = []
        for i in range(len(self.__data_lemmatized)):
            dist_p_tpc = self.get_topic_distribution(self.__data_lemmatized[i])
            for tpc in dist_p_tpc:
                endp_tpcs.append([i, tpc[0], tpc[1]])

        return endp_tpcs

    def cur_model_topic_words(self, t_num, top_words):
        topic = self.__current_model.show_topic(t_num, top_words)
        topic_words = [(t[0], t[1]) for t in topic]

        return topic_words

    def cur_model_topic_quantity(self):
        return self.__current_model.num_topics