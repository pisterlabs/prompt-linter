from collections import defaultdict

from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary
from gensim.models import AuthorTopicModel

import numpy as np

from scipy.spatial.distance import cdist

from sklearn.model_selection import train_test_split

from .utility import compose, hellinger


class ATMBuildHandler:
    '''
    ATMBuildHandler takes inputs to the author topic model, and builds
    out all resources nessicary to save and load for the ExpertTopicModel
    '''
    def __init__(self,
                 documents, attributions, topics, iterations,
                 DOC_KEY, USER_KEY, DOC_VALUE='TARGET',
                 minimum_attributions = 2, maximum_attributions = 50,
    ):
        attributions = self._remove_doubles(
            attributions,
            [DOC_KEY, USER_KEY]
        )

        attributions = self._bound_authors(
            attributions, minimum_attributions, maximum_attributions, USER_KEY
        )

        documents, attributions = self._author_document_downselect(
            documents, attributions, DOC_KEY
        )

        train, test, vocab = self._get_train_test_and_vocab(
            documents, DOC_VALUE
        )
        self.train, self.test, self.vocab = train, test, vocab

        train_attr = self._attribution_table(
            self.train, attributions, DOC_KEY, USER_KEY
        )
        test_attr = self._attribution_table(
            self.test, attributions, DOC_KEY, USER_KEY
        )
        self.train_attr, self.test_attr = train_attr, test_attr

        self.train_corpus = self._processed_to_bow(self.train[DOC_VALUE], self.vocab)
        self.test_corpus = self._processed_to_bow(self.train[DOC_VALUE], self.vocab)

        self.topics = topics
        self.iterations = iterations

        self.model = AuthorTopicModel(
            corpus=self.train_corpus, author2doc=self.train_attr,
            num_topics=self.topics, iterations=self.iterations,
        )

    @classmethod
    def _processed_to_bow(cls, corpus, vocab):
        return list(corpus.str.split().apply(vocab.doc2bow))

    @classmethod
    def _remove_doubles(cls, authors, columns):
        return authors.drop_duplicates(subset=columns)

    @classmethod
    def _bound_authors(cls, authors, minimum, maximum, USER_KEY):
        af = authors[USER_KEY].value_counts().reset_index()
        af.columns = ['col', 'count']

        idx = af['count'] > minimum
        af = af[idx]

        idx = af['count'] < maximum
        af = af[idx]

        result = authors[authors[USER_KEY].isin(af['col'])]
        return result

    @classmethod
    def _author_document_downselect(cls, documents, authors, DOC_KEY):

        idx = documents[DOC_KEY].isin(
            authors[DOC_KEY].unique()
        )
        documents = documents[idx]

        idx = authors[DOC_KEY].isin(
            documents[DOC_KEY].unique()
        )
        authors = authors[idx]
        return documents, authors

    @classmethod
    def _get_train_test_and_vocab(cls, documents, DOC_VALUE, test_size=.05, min_occurances=15):
        train, test = train_test_split(
            documents, test_size=test_size
        )
        train.reset_index(drop=True, inplace=True)
        test.reset_index(drop=True, inplace=True)

        vocab = Dictionary(
            train[DOC_VALUE].apply(str).str.split()
        )
        vocab.filter_extremes(no_below=min_occurances)

        return train, test, vocab

    @classmethod
    def _attribution_table(cls, documents, relevent_authors, DOC_KEY, USER_KEY):
        store = defaultdict(set)
        for idx, anomaly in documents.iterrows():
            authors_documents = relevent_authors[
                relevent_authors[DOC_KEY] == anomaly[DOC_KEY]
            ]

            for author in authors_documents[USER_KEY]:
                store[author].add(idx)

        return {k: list(v) for k, v in store.items()}

    def save(self, spec, dstpath):
        spec.save_atm(dstpath, self.model)
        spec.save_vocab(dstpath, self.vocab)
        spec.save_train(dstpath, self.train, self.train_attr)
        spec.save_test(dstpath, self.test, self.test_attr)


class ExpertTopicModel:
    '''
    Expert Topic Model provides a full interface to the learned
    Author Topic Model as specified by the `document_spec`. This handles
    all queries and supporting transformations against an input document.
    '''
    def __init__(self, document_spec, path):
        #document_spec = document_spec
        self.agglomerator = document_spec.AgglomerateProcessor()
        self.re_preprocessor = document_spec.RegexProcessor()
        self.stemmer = document_spec.load_stemmer(path)

        self.model = document_spec.load_atm(path)
        self.vocab = document_spec.load_vocab(path)

    def _process(self, document):
        helper = compose(
            self.agglomerator,
            self.re_preprocessor,
            self.stemmer.stem_unstem_document,
            str.split,
            self.vocab.doc2bow
        )
        return helper(document)

    def _get_document_topics(self, doc_bow):

        gamma_chunk, sstats = self.model.inference(
            chunk=[doc_bow], author2doc=dict(), doc2author=dict(),
            rhot=1.00,
            collect_sstats=True
        )

        return gamma_chunk

    @property
    def author_topic_vectors(self):
        author_topic_vectors = np.zeros(
            (self.model.num_authors, self.model.num_topics)
        )

        for i, author in enumerate(self.model.id2author.values()):
            idx, scores = zip(*self.model.get_author_topics(author))
            author_topic_vectors[i, idx] = scores

        return author_topic_vectors


    def _get_sorted_authors(self, doc_bow, metric):
        doc_vector = self._get_document_topics(doc_bow)

        author_scores = np.argsort(
            cdist(doc_vector, self.author_topic_vectors, metric=metric)
        )

        contenders = [
            self.model.id2author[idx]
            for idx in author_scores[0]
        ]

        return contenders

    def get_experts(self, document, metric=hellinger):
        bow = self._process(document)
        candidates = self._get_sorted_authors(bow, metric)

        return candidates
