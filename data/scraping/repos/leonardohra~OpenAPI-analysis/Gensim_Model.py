import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
from gensim.models.lsimodel import LsiModel
from gensim.models.nmf import Nmf

import numpy as np
import re, nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from string import ascii_lowercase

import csv
import spacy

import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

from pprint import pprint
from wordcloud import WordCloud

from CustomEnumerators import TopicModelingAlgorithm, CoherenceType


class Gensim_Model():

    __data_list = []
    __data_lemmatized = []
    __id_to_orig_text = {}
    __corpus = []
    __id2_words = None
    __optimal_models = {}
    __best_par = []
    __df_topic_sents_keywords = None
    __default_model = None
    __nlp = None

    def __init__(self, data_list, preprocessed_text = None):
        self.__data_list = data_list

        self.__nlp = spacy.load('en') # load model with shortcut link "en"
        print('Gensim model - spacy model loaded')
        """https://spacy.io/api/annotation"""
        self.__data_lemmatized = [self.__pre_process(dat) for dat in data_list]

        print('Gensim model - data preprocessed')

        self.__corpus = self.__vectorization(self.__data_lemmatized)
        print('Gensim model - The corpus has {0} documents'.format(len(self.__corpus)))

    def __sent_to_words(self, sentence):
        # deacc=True removes punctuations
        return gensim.utils.simple_preprocess(str(sentence), deacc=True)

    def __remove_stopwords(self, tokens):
        stop_words = stopwords.words('english')
        return [tk for tk in tokens if tk not in stop_words]

    def __lemmatization(self, text, deb = False):
        text_out = []
        doc = self.__nlp(" ".join(text))

        if(deb):
            print([(token.lemma_, token.pos_) for token in doc ])

        return [token.lemma_ for token in doc]

    '''def __lemmatization_bak(self, text, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'], deb = False):
        text_out = []
        doc = self.__nlp(" ".join(text))

        if(deb):
            print([(token.lemma_, token.pos_) for token in doc ])

        return [token.lemma_ for token in doc if token.pos_ in allowed_postags]'''

    def __vectorization(self, data_lemmatized):
        # Create Dictionary
        self.__id2_words = corpora.Dictionary(data_lemmatized)
        print('The dictionary has {0} tokens'.format(len(self.__id2_words.token2id)))

        # Create Corpus
        texts = data_lemmatized

        # Term Document Frequency
        corpus = [self.__id2_words.doc2bow(text) for text in texts]

        return corpus

    def __pre_process(self, text, deb=False):
        data_tkzed = list(self.__sent_to_words(text))
        data_stp = self.__remove_stopwords(data_tkzed)
        data_lemmatized = self.__lemmatization(data_stp)

        if(deb):
            print(data_tkzed)
            print(data_stp)
            print(data_lemmatized)

        return data_lemmatized

    def get_id2words(self):
        return self.__id2_words

    def get_corpus(self):
        return self.__corpus

    def id_to_endpoint(self, id):
        return self.__id_to_orig_text[id]

    def get_default_model(self):
        return self.__default_model

    def topics_for_new_text(self, text):
        text_id2w = self.__id2_words.doc2bow(text)
        text_to_tpcs = self.__default_model[text_id2w]
        qtt_tpcs = self.__default_model.num_topics

        id_to_tpc = []

        for j in range(qtt_tpcs):
            found = False
            for elem in text_to_tpcs:
                if(elem[0] == j):
                    id_to_tpc.append([123456789, j, elem[1]])
                    found = True
            if(not found):
                id_to_tpc.append([123456789, j, 0])

        df2 = pd.DataFrame(id_to_tpc, columns=['id_sentence', 'topic_num', 'score'])

        return df2


    def pre_process_external(self, text):
        lemmatized = self.__pre_process(text)
        return lemmatized

    def __calculate_coherence(self, model, topn, coherence_type=CoherenceType.C_V):
        coherence_model = CoherenceModel(model=model, texts=self.__data_lemmatized, corpus=self.__corpus, dictionary=self.__id2_words, coherence=str(coherence_type), topn=topn)
        coherence = coherence_model.get_coherence()
        return coherence

    def __create_model(self, algo, topic_qtt):
        model = None
        if(algo == TopicModelingAlgorithm.LDA):
            model = LdaModel(corpus=self.__corpus, num_topics=topic_qtt, id2word=self.__id2_words, random_state=1)
        elif(algo == TopicModelingAlgorithm.LSA):
            model = LsiModel(corpus=self.__corpus, num_topics=topic_qtt, id2word=self.__id2_words)
        elif(algo == TopicModelingAlgorithm.NMF):
            model = Nmf(corpus=self.__corpus, num_topics=topic_qtt, random_state=1)

        return model

    def set_default_model(self, algo, topic_qtt):
        self.__default_model = self.__create_model(algo, topic_qtt)


    def model_construction_evaluation_topn(self, out_folder, out_file, algorithms = [TopicModelingAlgorithm.LDA, TopicModelingAlgorithm.LSA, TopicModelingAlgorithm.NMF], coherence_types = [ CoherenceType.U_MASS, CoherenceType.C_V, CoherenceType.C_UCI, CoherenceType.C_NPMI ], qtt_topics = range(1,21), topns = range(5,11)):
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
                for coherence in coherence_types:
                    for topn in topns:
                        model = self.__create_model(algorithm, qtt)
                        topics = model.show_topics(formatted=False, num_words=topn)
                        c = self.__calculate_coherence(model, topn, coherence)

                        all_values[coherence].append([algorithm, qtt, topn, c])

                        print('{0}, {1}, {2}, {3}, {4}'.format(algorithm, coherence, qtt, topn, c))

                        for topic in topics:
                            algo = algorithm
                            qtt = qtt
                            topic_n = topic[0]
                            words_join = []
                            top_words = topn

                            if algo == TopicModelingAlgorithm.NMF:
                                words = ['{:.2f} * {}'.format(round(word[1], 2), self.__id2_words[int(word[0])]) for word in topic[1]]
                                words_join = ' + '.join(words)
                            else:
                                words = ['{:.2f} * {}'.format(round(word[1], 2), word[0]) for word in topic[1]]
                                words_join = ' + '.join(words)

                            topics_all.append([algorithm.simple_name(), qtt, top_words, topic_n, words_join])

                        #print('Done for {} topics'.format(qtt))

                        if(c > best_par[coherence][-1]):
                            #print('\n\n\nNew best parameters! \nAlgorithm: {0}\nCoherence Type: {1}\nCoherence value: {2} \nNumber of Topics: {3}\n'.format(str(algorithm), str(coherence), c, qtt))
                            best_par[coherence] = [algorithm, qtt, c]

        self.__best_par = best_par
        #print(best_par)
        #print(all_values)

        w = csv.writer(open(out_folder + out_file + '.csv', 'w'))
        w2 = csv.writer(open(out_folder + out_file + '_topics.csv', 'w'))

        for values in all_values[CoherenceType.C_V]:
            w.writerow(values)

        for topics in topics_all:
            w2.writerow(topics)

    def endpoint_in_topics_dataset(self):
        #id of endpoint, topic number, % in topic
        id_to_tpc = []
        qtt = self.__default_model.num_topics
        lim = len(self.__corpus)
        for i in range(lim):
            tpcs = self.__default_model[self.__corpus[i]]
            for j in range(qtt):
                found = False
                for elem in tpcs:
                    if(elem[0] == j):
                        id_to_tpc.append([i, j, elem[1]])
                        found = True
                if(not found):
                    id_to_tpc.append([i, j, 0])
            if(i % 1000 == 0):
                print('{}/{}'.format(i, lim))
        return id_to_tpc

    def __format_topics_sentences(self, coh_type = CoherenceType.C_V):
        ldamodel=self.__optimal_models[coh_type]
        corpus=self.__corpus
        texts=self.__data_list

        # Init output
        sent_topics_df = pd.DataFrame()

        # Get main topic in each document
        for i, row in enumerate(ldamodel[corpus]):
            row = sorted(row, key=lambda x: (x[1]), reverse=True)
            # Get the Dominant topic, Perc Contribution and Keywords for each document
            for j, (topic_num, prop_topic) in enumerate(row):
                if j == 0:  # => dominant topic
                    wp = ldamodel.show_topic(topic_num)
                    topic_keywords = ", ".join([word for word, prop in wp])
                    sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,2), topic_keywords]), ignore_index=True)
                else:
                    break
        sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

        # Add original text to the end of the output
        contents = pd.Series(texts)
        sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
        return(sent_topics_df)
