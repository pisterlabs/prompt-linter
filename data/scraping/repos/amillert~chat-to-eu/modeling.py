import os
from pprint import pprint #as print

import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
from gensim.test.utils import datapath

import spacy

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class Modeler:
    def __init__(self, tmp_path, paragraph_tokens=None, load=True):
        self.tmp_path = tmp_path
        if load == True:
            self.lda_model = self.read_model()
        else:
            self.paras = paragraph_tokens
            lemms = self.lemmatize()
            # word2idx = {word: idx for (idx, word) in enumerate(vocab)}
            # idx2word = {idx: word for (word, idx) in word2idx.items()}
            idx2word = corpora.Dictionary(lemms)
            corpus = [idx2word.doc2bow(text) for text in lemms]
            self.lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=idx2word, num_topics=20, update_every=1, chunksize=100, passes=20, alpha='auto', per_word_topics=True)
            self.save_model()
            # lda_model.update(other_corpus)
            # vector = lda_model[unseen_doc]

    def lemmatize(self):
        bigram = gensim.models.Phrases(self.paras, min_count=3, threshold=65) # higher threshold fewer phrases.
        bigram_mod = gensim.models.phrases.Phraser(bigram)
        # >> pyon3 -m spacy download en
        nlp = spacy.load('en', disable=['parser', 'ner'])
        bigrams = [bigram_mod[para] for para in self.paras]
        allowed_postags = ["NOUN", "ADJ", "VERB", "ADV", "NUM"]
        return [[token.lemma_ for token in nlp(" ".join(para)) if token.pos_ in allowed_postags] for para in bigrams]
    
    def save_model(self):
        self.lda_model.save(self.tmp_path + "lda")

    def read_model(self):
        return gensim.models.ldamodel.LdaModel.load(self.tmp_path + "lda")
