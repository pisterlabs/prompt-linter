import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel


class Evaluator:

    def computeTFIDF(self, documents):
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(documents)
        feature_names = vectorizer.get_feature_names()
        dense = vectors.todense()
        denselist = dense.tolist()
        df = pd.DataFrame(denselist, columns=feature_names)
        df.index = ([f"TFIDF_{i}" for i in range(1, len(documents) + 1)])

        return df

    def countvectorizer(self, documents):
        countvectorizer = CountVectorizer()
        X = countvectorizer.fit_transform(documents)
        result = X.toarray()
        return result

    def similarity(self, documents):
        similarities = {}
        for i, document in enumerate(documents, 1):
            doc = nlp(document)
            sim = []
            for document_ in documents:
                sim.append(doc.similarity(nlp(document_)))
            similarities[f"Doc_{i}"] = sim
        df = pd.DataFrame(similarities, index=[f"Doc_{i}" for i in range(1, len(documents) + 1)])
        return df

    def display_lda(self, lda_model, corpus, id2word):
        return
        pyLDAvis.enable_notebook()
        vis = gensimvis.prepare(lda_model, corpus, id2word, mds="mmds", R=30)
        return vis

    def LDA(self, data_words):
        id2word = corpora.Dictionary(data_words)
        corpus = []

        for text in data_words:
            new = id2word.doc2bow(text)
            corpus.append(new)
        num_topic = len(corpus)
        lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                    id2word=id2word,
                                                    num_topics=num_topic,
                                                    random_state=100,
                                                    update_every=1,
                                                    chunksize=100,
                                                    passes=10,
                                                    eta=[.01] * len(id2word.keys()),
                                                    alpha=[.01] * num_topic)
        return lda_model, self.display_lda(lda_model, corpus, id2word)
