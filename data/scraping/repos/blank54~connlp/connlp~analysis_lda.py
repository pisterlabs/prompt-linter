#!/usr/bin/env python -W ignore::DeprecationWarning
# -*- coding: utf-8 -*-

# Configuration
from collections import defaultdict

import gensim.corpora as corpora
from gensim.models import CoherenceModel
from gensim.models.ldamodel import LdaModel

import pyLDAvis.gensim


class TopicModel:
    '''
    A class for topic modeling based on LDA.
    Refer to: https://radimrehurek.com/gensim/models/ldamodel.html

    Attributes
    ----------
    docs : dict
        | A dict of docs, of which key is tag and value is tokenized text.
    num_topics : int
        | The number of topics of the docs.

    Methods
    -------
    learn
        | Trains LDA model with given parameters.
        | Detail information for each parameter is provided in gensim website.
    assign
        | Assigns topic for each doc.
    '''

    def __init__(self, docs, num_topics):
        self.docs = docs
        self.id2word = corpora.Dictionary(self.docs.values())

        self.model = ''
        self.coherence = ''

        self.num_topics = num_topics
        self.docs_for_lda = [self.id2word.doc2bow(text) for text in self.docs.values()]
        
        self.tag2topic = defaultdict(int)
        self.topic2tag = defaultdict(list)

    def learn(self, **kwargs):
        parameters = kwargs.get('parameters', {})
        self.model = LdaModel(corpus=self.docs_for_lda,
                              id2word=self.id2word,
                              num_topics=self.num_topics,
                              iterations=parameters.get('iterations', 100),
                              update_every=parameters.get('update_every', 1),
                              chunksize=parameters.get('chunksize', 100),
                              passes=parameters.get('passes', 10),
                              alpha=parameters.get('alpha', 0.5),
                              eta=parameters.get('eta', 0.5),
                              )

        self.__calculate_coherence()

        # print('Learn LDA Model')
        # print('  | # of docs  : {:,}'.format(len(self.docs)))
        # print('  | # of topics: {:,}'.format(self.num_topics))

    def __calculate_coherence(self):
        coherence_model = CoherenceModel(model=self.model,
                                         texts=self.docs.values(),
                                         dictionary=self.id2word)
        self.coherence = coherence_model.get_coherence()

    def assign(self):
        doc2topic = self.model[self.docs_for_lda]
        
        for idx, tag in enumerate(self.docs):
            topic_id = sorted(doc2topic[idx], key=lambda x:x[1], reverse=True)[0][0]

            self.tag2topic[tag] = topic_id
            self.topic2tag[topic_id].append(tag)

    def visualize(self):
        pyLDAvis.gensim.prepare(self.model, self.docs_for_lda, self.id2word)