import gensim, logging, pickle
import pandas as pd
import numpy as np
from octis.models.NeuralLDA import NeuralLDA
from octis.dataset.dataset import Dataset
import os
import string
from octis.preprocessing.preprocessing import Preprocessing
from octis.evaluation_metrics.diversity_metrics import TopicDiversity
from octis.evaluation_metrics.coherence_metrics import Coherence

from sklearn.feature_extraction.text import CountVectorizer

import gensim
from gensim.models.coherencemodel import CoherenceModel
import gensim.corpora as corpora

from bertopic import BERTopic

import nltk

stop_words = nltk.corpus.stopwords.words('english')

import params
from .mdl import AbstractAspectModel


class Neural(AbstractAspectModel):
    def __init__(self, reviews, naspects, no_extremes, output):
        super().__init__(reviews, naspects, no_extremes, output)

    def load(self):
        self.mdl = BERTopic.load(f'{self.path}model')
        # assert self.mdl.num_topics == self.naspects
        self.dict = gensim.corpora.Dictionary.load(f'{self.path}model.dict')
        with open(f'{self.path}model.perf.cas', 'rb') as f: self.cas = pickle.load(f)
        with open(f'{self.path}model.perf.perplexity', 'rb') as f: self.perplexity = pickle.load(f)
    # def preprocess(doctype, reviews):
    #     reviews_ = [s for r in reviews for s in r.sentences]

    def train(self, doctype, cores, iter, seed):

        # model = NeuralLDA(num_topics=self.naspects, batch_size=params.iter_c)
        # reviews_ = super().preprocess(doctype, self.reviews)
        # reviews_ = [' '.join(text) for text in reviews_]
        # train_tag = ['train' for r in reviews_]
        # vectorizer = CountVectorizer()
        # vectorizer.fit_transform(reviews_)
        # # Get the list of unique words
        # self.dict = vectorizer.get_feature_names()
        # model_path = self.path[:self.path.rfind("/")]
        # with open(f'{model_path}/vocabulary.txt', "w", encoding="utf-8") as file:
        #     for item in self.dict:
        #         file.write("%s\n" % item)
        # with open(f'{model_path}/corpus.tsv', "w", encoding="utf-8") as outfile:
        #     for i in range(len(reviews_)):
        #         outfile.write("{}\t{}\n".format(reviews_[i], train_tag[i]))
        # dataset = Dataset()
        # dataset.load_custom_dataset_from_folder(f'{model_path}')
        # self.mdl = model.train_model(dataset)

        # npmi = Coherence(texts=dataset.get_corpus(), topk=params.nwords, measure='u_mass')
        # self.cas = npmi.score(self.mdl)
        # self.perplexity = 0
        # pd.to_pickle(self.dict, f'{self.path}model.dict.pkl')
        # pd.to_pickle(self.mdl, f'{self.path}model.pkl')

        reviews_ = super().preprocess(doctype, self.reviews)
        doc = [' '.join(text) for text in reviews_]

        self.mdl = BERTopic(nr_topics=None, top_n_words=params.nwords, calculate_probabilities=True)
        topics, probabilities = self.mdl.fit_transform(doc)
        # self.mdl.get_topic_info()
        aspects, probs = self.get_aspects(params.nwords)

        # # [-inf, 0]: close to zero, the better
        self.dict = gensim.corpora.Dictionary(reviews_)
        # if self.no_extremes: self.dict.filter_extremes(no_below=self.no_extremes['no_below'], no_above=self.no_extremes['no_above'], keep_n=100000)
        # if self.no_extremes: self.dict.filter_extremes(keep_n=100000)
        # self.dict.compactify()

        corpus = [self.dict.doc2bow(doc) for doc in reviews_]
        coherence_model = CoherenceModel(topics=aspects, texts=reviews_, corpus=corpus, dictionary=self.dict, coherence='u_mass')
        self.cas = coherence_model.get_coherence()
        log_perplexity = -1 * np.mean(np.log(np.sum(probabilities, axis=0)))
        self.perplexity = np.exp(log_perplexity)

        self.dict.save(f'{self.path}model.dict')
        self.mdl.save(f'{self.path}model')
        with open(f'{self.path}model.perf.cas', 'wb') as f:
            pickle.dump(self.cas, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f'{self.path}model.perf.perplexity', 'wb') as f:
            pickle.dump(self.perplexity, f, protocol=pickle.HIGHEST_PROTOCOL)

    def get_aspects(self, nwords):
        words = []
        probs = []
        for n in range(-1, len(self.mdl.topic_representations_)-1):

            topic_list = self.mdl.get_topic(n)
            word_list_per_topic = []
            probs_list_per_topic = []
            for t in topic_list:
                word_list_per_topic.append(t[0])
                probs_list_per_topic.append(t[1])
            words.append(word_list_per_topic)
            probs.append(probs_list_per_topic)
        return words, probs

    def show_topic(self, topic_id, nwords):
        topic_list = self.mdl.get_topic(topic_id)
        word_list_per_topic = []
        probs_list_per_topic = []
        for t in topic_list:
            word_list_per_topic.append(t[0])
            probs_list_per_topic.append(t[1])
        return list(zip(word_list_per_topic, probs_list_per_topic))

    def infer(self, doctype, review):
        review_aspects = []
        review_ = super().preprocess(doctype, [review])
        doc = [' '.join(text) for text in review_]
        if len(doc) == 0:
            return []
        else:
            p = self.mdl.transform(doc)
            for prob in p[1]:
                r_list = []
                for i in range(len(list(prob))):
                    r_list.append((i, prob[i]))
                review_aspects.append(r_list)
            return review_aspects

            # print(sum(list(prob)))
        # for r in doc:
        #     # print(self.mdl.transform(r))
        #     pp = self.mdl.transform([r])
        #     print(pp)
        #     review_aspects.append([(pp[0][0], pp[1][0])])

