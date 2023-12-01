import sys
import spacy
import pickle
import numpy as np
from nltk.corpus import stopwords
from gensim.corpora import Dictionary
from sklearn.linear_model import LinearRegression
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.wrappers import LdaMallet
import logging


class Serialization:
    @staticmethod
    def save_obj(obj, name):
        """
        serialization of an object
        :param obj: object to serialize
        :param name: file name to store the object
        """
        with open('pickle/' + name + '.pkl', 'wb') as fout:
            pickle.dump(obj, fout, pickle.HIGHEST_PROTOCOL)
        # end with
    # end def

    @staticmethod
    def load_obj(name):
        """
        de-serialization of an object
        :param name: file name to load the object from
        """
        with open('pickle/' + name + '.pkl', 'rb') as fin:
            return pickle.load(fin)
        # end with
    # end def

# end class


class DataUtils:

    def __init__(self):
        self.ALLOWED_POSTAGS = ['NOUN', 'ADJ', 'VERB', 'ADV']
        self.MAX_WORD_RANK = 10000  # based on Wikipedia word frequencies
        self.MIN_WORD_RANK = 300  # based on Wikipedia word frequencies
        self.MIN_POST_LENGTH = 50  # after cleanup

        self.nlp = spacy.load('en_core_web_lg', disable=['tokenizer', 'parser', 'ner'])
        self.word_ranks = Serialization.load_obj('dict.word.ranks')
        self.stop_words = stopwords.words('english')

    # end def

    def lemmatization(self, text):
        """https://spacy.io/api/annotation"""
        output = list()
        for token in self.nlp(text):
            if token.pos_ not in self.ALLOWED_POSTAGS: continue
            output.append(token.lemma_)
        # end for
        return output
    # end def

    def load_and_clean_data(self, filename):
        clean_data = list()
        data = Serialization.load_obj(filename)
        for week in range(len(data)):
            if week not in data: continue

            print('processing data for week', week)
            for i, post in enumerate(data[week]):
                clean = self.extract_post_clean_data(post)
                if len(clean) < self.MIN_POST_LENGTH: continue
                clean_data.append(clean)
            # end for
        # end for

        print('extracted', len(clean_data), 'posts')
        return clean_data

    # end def

    def extract_post_clean_data(self, post):
        clean_post = list()
        for word in self.lemmatization(post):
            # exclude stopwords and words with atypical length
            if word in self.stop_words or len(word) not in range(4, 15): continue
            # exclude too common or too rare words
            if self.word_ranks.get(word, 0) < self.MIN_WORD_RANK or \
                self.word_ranks.get(word, sys.maxsize) > self.MAX_WORD_RANK: continue

            clean_post.append(word)
        # end for
        return clean_post

    # end def

# end class


class TopicModeling():

    MALLET_PATH = 'mallet-2.0.8/bin/mallet'

    @staticmethod
    def find_best_number_of_topics(data):
        dictionary = Dictionary(data)
        corpus = [dictionary.doc2bow(text) for text in data]

        scores = dict()
        for topics in range(2, 10, 1):
            print('performing topic modeling with', topics, 'topics')
            ldamodel = LdaMallet(TopicModeling.MALLET_PATH, corpus=corpus, num_topics=topics, id2word=dictionary)
            coherence_model = CoherenceModel(model=ldamodel, texts=data, coherence='c_v')
            coherence = coherence_model.get_coherence()
            scores[topics] = coherence
        # end for

        print('coherence scores: the higher, the better:', scores)
    # end def

    @staticmethod
    def generate_topics(data, topics, gender):
        dictionary = Dictionary(data)
        corpus = [dictionary.doc2bow(text) for text in data]
        print('performing topic modeling with', topics, 'topics')
        ldamodel = LdaMallet(TopicModeling.MALLET_PATH, corpus=corpus, num_topics=topics, id2word=dictionary)
        ldamodel.save('ldamodel.' + gender + '.' + str(topics))

    # end def

# end class


def explore_temporal_trends(topics, gender):
    model = LdaMallet.load('ldamodel.' + gender + '.' + str(topics))
    topic_words = model.show_topics(num_topics=topics, num_words=100, formatted=False)
    words_only = [(tp[0], [wd[0] for wd in tp[1]]) for tp in topic_words]

    words = list()
    for tp in words_only: words.extend(tp[1])

    weekly_stats = dict()
    data = Serialization.load_obj('week2comments.' + gender)
    for week in range(len(data)):
        if week not in data: continue
        print('processing data for week', week)
        week_data = ' '.join(data[week]).lower()
        total = len(week_data.split())

        for word in set(words):
            current_word_stats = weekly_stats.get(word, list())
            current_word_stats.append(float(week_data.split().count(word))/total)
            weekly_stats[word] = current_word_stats
        # end for
    # end for

    for word in weekly_stats:
        y = np.array(weekly_stats[word])
        x = np.array([i for i in range(len(weekly_stats[word]))]).reshape(-1, 1)
        model = LinearRegression().fit(x, y)
        print(word, model.coef_[0])
    # end for

# end def


# todo: a reasonably good tutorial on topic modeling is here -
# https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/

# todo: the code assumes mallet topic modeling project in <current_dir>/mallet-2.0.8/
# todo: can download from http://mallet.cs.umass.edu/topics.php
if __name__ == '__main__':
    gender = 'M'

    utils = DataUtils()
    # the code assumes week2comments pkl file (e.g., week2comments.M.pkl) in pickle/
    # the file is a dictionary pickle object: week -> list of post and comments for that week
    data = utils.load_and_clean_data('week2comments.' + gender)
    Serialization.save_obj(data, 'clean.data.' + gender)

    modeling = TopicModeling()
    data = Serialization.load_obj('clean.data.' + gender)
    modeling.find_best_number_of_topics(data)  # todo: inspect coherence scores and decide on topics number

    topics = 8  # based on manual inspection in the previous step
    modeling.generate_topics(data, topics, gender)

    explore_temporal_trends(topics, gender)


# end if
