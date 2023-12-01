
from gensim import corpora
from gensim.corpora import Dictionary
from gensim.models import LsiModel, TfidfModel
from gensim.models.coherencemodel import CoherenceModel
from keras.preprocessing.sequence import pad_sequences
from time import time
from embedding_methods_old.one_hot import OneHot
from embedding_methods_old.base_embedding import BaseEmbedding
import numpy as np


class LSAEmbedding(BaseEmbedding):
    def __init__(self, embedding_dictionary_file, word_to_index_file, docs_tokens, doc_len, word_len, iters):
        self.time = 0.
        if embedding_dictionary_file is not None and word_to_index_file is not None:
            super(LSAEmbedding, self).get_from_files(embedding_dictionary_file, word_to_index_file, doc_len, self)
        else:
            self.time = time()
            word_dictionary = Dictionary(docs_tokens)
            word_to_index = word_dictionary.token2id
            docs_term_matrix = [word_dictionary.doc2bow(tokens) for tokens in docs_tokens]
            tfidfmodel = TfidfModel(docs_term_matrix, id2word=word_dictionary)
            corpus = [tfidfmodel[doc] for doc in docs_term_matrix]
            lsamodel = LsiModel(corpus, num_topics=word_len, id2word=word_dictionary, power_iters=iters)
            self.time = time() - self.time

            embedding_matrix = lsamodel.get_topics().transpose()
            embedding_dictionary = {}
            embedding_dim = None
            for word, i in word_to_index.items():
                embedding_dictionary[word] = embedding_matrix[i]
                if embedding_dim is None:
                    embedding_dim = len(embedding_matrix[i])

            # print(lsamodel.print_topics(num_topics=number_of_topics, num_words=words))
            # one_hot = OneHot(docs_tokens, max_doc_len=self.doc_len)
            # word_to_index = one_hot.get_word_indexes()
            super(LSAEmbedding,self).get_from_data(embedding_dictionary, embedding_dim, word_to_index, doc_len, self)

        self.name = 'lsa'
        self.iters = iters



        # word_dictionary = Dictionary(self.docs_tokens)
        # if "<<UNKNOWN>>" in word_dictionary.token2id:
        #     unknown_index = word_dictionary.token2id["<<UNKNOWN>>"]
        #     word_dictionary.filter_tokens(bad_ids=[unknown_index])
        # word_dictionary.merge_with(Dictionary([['<<UNKNOWN>>']]))
        # self.unknown_index = word_dictionary.token2id["<<UNKNOWN>>"]

        # docs_term_matrix = [word_dictionary.doc2bow(tokens) for tokens in self.docs_tokens]
        # tfidfmodel = TfidfModel(docs_term_matrix, id2word=word_dictionary)
        # corpus = [tfidfmodel[doc] for doc in docs_term_matrix]
        # # generate LSA model
        # lsamodel = LsiModel(corpus, num_topics=self.word_len, id2word=word_dictionary, power_iters=iters)
        # self.time = time() - self.time

        # self.__word_dictionary = word_dictionary
        # self.word_to_index_dic = word_dictionary.token2id
        # self.index_to_word_dic = {index: word for word, index in self.word_to_index_dic.items()}

        # print(lsamodel.print_topics(num_topics=number_of_topics, num_words=words))
        # embedding_matrix = lsamodel.get_topics().transpose()
        # self.embedding_dictionary = self.get_embedding_dictionary(embedding_matrix, word_dictionary.token2id)
        # self.get_embedding_matrix()

    # def get_embedding_dictionary(self, embedding_matrix, word_to_index):
    #     embedding_dictionary = {}
    #     for word, i in word_to_index.items():
    #         embedding_dictionary[word] = embedding_matrix[i]
    #     return embedding_dictionary
    #     # embedding_dictionary = {}
    #     # for i, vector in enumerate(self.embedding_matrix):
    #     #     word = self.index_to_word_dic.get(i, None)
    #     #     if word is not None:
    #     #         embedding_dictionary[word] = vector
    #     # return embedding_dictionary

    #
    # def get_embedding_matrix(self):
    #     self.one_hot = OneHot(self.docs_tokens, max_doc_len=self.doc_len)
    #     self.word_to_index_dic = self.one_hot.get_word_indexes()
    #     word_index = self.word_to_index_dic
    #     embedding_matrix = np.zeros((len(word_index)+1, self.embedding_dim))
    #     for word, i in word_index.items():
    #     # for i in range(1, self.word_num):
    #     #     word = self.one_hot.tokenizer.index_word[i]
    #         embedding_vector = self.embedding_dictionary.get(word, None)
    #         if embedding_vector is not None:
    #             # words not found in embedding index will be all-zeros.
    #             embedding_matrix[i] = embedding_vector
    #     self.embedding_matrix = embedding_matrix

    # def get_sequence(self, tokens, unknown_word_index=None):
    #     index_of_unknown = 0
    #     return pad_sequences(
    #         [self.__word_dictionary.doc2idx(tokens, unknown_word_index=self.unknown_index)],
    #         self.doc_len
    #     )[0]
    #
    # def get_docs_sequences(self, docs_tokens):
    #     try:
    #         return pad_sequences(
    #             [self.__word_dictionary.doc2idx(tokens, unknown_word_index=self.unknown_index) for tokens in docs_tokens],
    #             self.doc_len
    #         )
    #     except:
    #         return None
    #
    # def get_tokens(self, sequence):
    #     raise NotImplementedError()
    #     # return self.tokenizer.sequences_to_texts([sequence])[0]
    #
    # def get_docs_tokens(self, docs_sequences):
    #     raise NotImplementedError()
    #     # return self.tokenizer.sequences_to_texts(docs_sequences)

    # def get_sequence(self, tokens, unknown_word_index=None):
    #     return self.one_hot.get_sequence(tokens)
    #
    # def get_docs_sequences(self, docs_tokens):
    #     return self.one_hot.get_docs_sequences(docs_tokens)
    #
    # def get_tokens(self, sequence):
    #     return self.one_hot.get_tokens(sequence)
    #
    # def get_docs_tokens(self, docs_sequences):
    #     return self.one_hot.get_docs_tokens(docs_sequences)
