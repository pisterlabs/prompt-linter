from gensim.models.coherencemodel import CoherenceModel
from m2lib.pickler.picklable import PickleDef, Picklable
from m2lib.model.gsdmmSttmModel import GSDMM, GSDMMModelStore
from m2lib.model.ldaModel import LDAModel, LDATFIDModel, LDATFIDModelStore, LDAModelStore
from m2lib.pickler.picklable import Picklable, PickleDef
from m2lib.featureizers.preprocessor import Preprocessor, PreprocessorStore
from m2lib.readers.readdata import Read
from m2lib.featureizers.bowfeature import BOWFeature, BOWFeatureStore
import pyLDAvis
import pyLDAvis.gensim
from configurations import HTML_DIR
import time
import os
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class TopicsCharting(Picklable):
    def __init__(self):
        self.metricsStore = None
        pd = PickleDef(self)
        self.pickle_kwargs = pd()
        super().__init__(**self.pickle_kwargs)

    def pipeline(self, K_ = [1,2,3,4,5,6,7,8,9,10,15,20], p= 0.2):
        # setup data
        prepro = PreprocessorStore()
        corpus = prepro.corpus_
        _, corpus_ = train_test_split(corpus, test_size=p, random_state=42)
        bow = BOWFeature()
        bow.pipeline(corpus_)
        train_corpus = bow.corpus_
        train_dict = bow.dictionary

        # classes to build metrics and store
        metrics = Metrics()
        metricsStore = MetricsStore()

        for K in tqdm(K_):
            # model setup
            ldaModel = LDAModel()
            ldaModel.lda_args['num_topics'] = K
            ldaModel.train_model(train_corpus, train_dict)

            # get Metrics and store them
            metrics.pipeline(ldaModel.model, bow.corpus_, K)
            metricsStore.coherence.append(metrics.coherence)
            metricsStore.perplexity.append(metrics.perplexity)
            metricsStore.K.append(K)
            metricsStore.save()

        self.metricsStore = metricsStore
        self.save()
        return metricsStore

    def make_visualization(self):
        fig, ax = plt.subplots(2,1)
        x = self.metricsStore.K
        y = self.metricsStore.coherence
        y2 = self.metricsStore.perplexity

        fig.suptitle('Coherence and Perplexity Against Topics')

        ax[0].plot(x, y)
        ax[1].plot(x, y2)

        ax[0].set_title('Coherence')
        ax[1].set_title('Perplexity')

        ax[0].set(xlabel='K Topics', ylabel='Coherence')
        ax[1].set(xlabel='K Topics', ylabel='Perplexity')

        plt.show()

    def save(self):
        super().save()

    def load(self):
        super().load()

class MetricsStore(Picklable):
    def __init__(self):
        self.coherence = []
        self.perplexity = []
        self.K = []
        pd = PickleDef(self)
        self.pickle_kwargs = pd()
        super().__init__(**self.pickle_kwargs)

    def save(self):
        super().save()

    def load(self):
        super().load()

class Metrics(Picklable):
    def __init__(self):
        self.K = None
        self.coherence = None
        self.perplexity = None
        pd = PickleDef(self)
        self.pickle_kwargs = pd()
        super().__init__(**self.pickle_kwargs)

    def pipeline(self, model, corpus, K):
        # calc coherence
        cm = CoherenceModel(model=model, corpus=corpus, coherence='u_mass')
        coherence = cm.get_coherence()
        self.coherence = coherence

        # calc perplexity
        perplexity = model.log_perplexity(corpus)
        self.perplexity = perplexity

        self.K = K

    def save(self):
        super().save()

    def load(self):
        super().load()


if __name__ == '__main__':
    # ldaStore = LDAModelStore()
    # bowStore = BOWFeatureStore()
    # metrics = Metrics()
    # metricsStore = MetricsStore()
    # metrics.pipeline(ldaStore.model, bowStore.corpus_, 10)
    # metricsStore.coherence.append(metrics.coherence)
    # metricsStore.perplexity.append(metrics.perplexity)
    # metricsStore.K.append(10)
    # metricsStore.save()
    topicsChart = TopicsCharting()
    topicsChart.pipeline()
    topicsChart.make_visualization()


