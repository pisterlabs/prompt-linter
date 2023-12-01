# Ian Hay - 2023-03-14

import util as util
import numpy as np

from sklearn.cluster import MiniBatchKMeans as km
from sklearn.decomposition import LatentDirichletAllocation as lda
from sklearn.decomposition import NMF as nmf
from hdbscan import HDBSCAN # TODO
from gensim.models import ldamulticore
from gensim.models.coherencemodel import CoherenceModel
from gensim.parsing.preprocessing import preprocess_string, strip_punctuation, strip_numeric
from gensim import corpora
from gensim import matutils
from timeit import default_timer
from sklearn.base import TransformerMixin

from model.basemodel import BaseModel # local file

### --- abstract class --- ###

class ClusteringModel(BaseModel, TransformerMixin):

    def __init__(self):
        super().__init__()

    def fit(self, x, y=None):
        """
        Takes in and trains on the data `x` to return desired features `y`.
        Parameters:
            - x : ndarray[float] : 2d array of datapoints n samples by d features
            - y : ndarray[int] : topic prediction of n samples by c classes
        """
        util.raiseNotDefined()

    def transform(self, x):
        """
        For lowercase ngrams, featurizes them based on the trained model.
        Parameters:
            - x : ndarray[float] : list of datapoints n samples by d features
        Returns:
            - ypred : ndarray[int] : topic prediction of n samples bc y classes
        """
        util.raiseNotDefined()

    def perplexity(self, x):
        util.raiseNotDefined()

    def coherence(self, x):
        util.raiseNotDefined()


    # TODO
    def save(self):
        """
        Saves this model and any associated experiments to a .txt file.\n
        Returns:
            - filename : str : the filename this model's .txt file was saved to
        """
        util.raiseNotDefined()

    # TODO: for view
    def __dict__(self):
        """
        Represents this model as a string.\n
        Returns:
            - tostring : str : string representation of this model.
        """
        util.raiseNotDefined()



### -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ###


class SKLearnLDA(ClusteringModel):

    def __init__(self, nClasses, nJobs=14, batchSize=512, maxIter=10):
        super().__init__()
        self.nClasses = nClasses
        self.nJobs = nJobs
        self.batchSize = batchSize
        self.maxIter = maxIter


    def fit(self, x, y=None, verbose=False):
        self.model = lda(n_components=self.nClasses, batch_size=self.batchSize, max_iter=self.maxIter, n_jobs=self.nJobs)
        self.model.fit(x)
        return self


    def predict(self, x, y=None, verbose=False):
        output = self.model.transform(x)
        pred = util.getTopPrediction(output)
        return pred
    

    def perplexity(self, x):
        return self.model.perplexity(x)
    

    def coherence(self, x, nTop=10):
        # takes in the raw documents - TODO change this (make x's consistent)
        # TODO make this less bad

        # https://stackoverflow.com/questions/60613532/how-do-i-calculate-the-coherence-score-of-an-sklearn-lda-model
        topics = self.model.components_

        n_top_words = nTop
        # id2word = dict([(i, s) for i, s in enumerate(vocab)]) # do we need this?
        # corpus = matutils.Sparse2Corpus(x.T)

        _dict = corpora.dictionary.Dictionary(x)
        featnames = [_dict[i] for i in range(len(_dict))]
        corpus = [_dict.doc2bow(text) for text in x]

        topWords = []
        for _topic in topics:
            topWords.append([featnames[i] for i in _topic.argsort()[:-n_top_words - 1:-1]])
        cm = CoherenceModel(topics=topWords, texts=x, dictionary=_dict, topn=nTop, coherence='u_mass')
        return cm.get_coherence()
    

    def print_topics(self, vocab, nTopics=None, nTopWords=10, verbose=True):
        # https://stackoverflow.com/questions/44208501/getting-topic-word-distribution-from-lda-in-scikit-learn
        if (nTopics is None):
            nTopics = self.nClasses
        if (nTopics==0):
            return
        topicWords = {}
        for topic, comp in enumerate(self.model.components_):
            wordIdx = np.argsort(comp)[::-1][:nTopWords]
            topicWords[topic] = [vocab[i] for i in wordIdx]
        if verbose:
            for topic, words in topicWords.items():
                print('Topic: %d' % topic)
                print('  %s' % ', '.join(words))
        return topicWords


### -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ###


class GensimLDA(ClusteringModel):
    # TODO this can potentially be done better using a Gensim pipeline
    # instead of feeding in an SKLearn pipeline and converting on the fly

    def __init__(self, nClasses, vocab, nJobs=14, batchSize=512, maxIter=10,):
        super().__init__()
        self.nClasses = nClasses
        self.vocab = vocab
        self.nJobs = nJobs
        self.batchSize = batchSize
        self.maxIter = maxIter

    def fit(self, x, y=None, verbose=False):
        # needs a sparse array

        # https://gist.github.com/aronwc/8248457
        start = default_timer()
        id2word = dict([(i, s) for i, s in enumerate(self.vocab)])
        _dictTime = default_timer() - start
        if verbose: print(f"id2word dict creation: {_dictTime:.3f}")

        self.trainCorpus = matutils.Sparse2Corpus(x.T)
        _corpusTime = default_timer() - start - _dictTime
        if verbose: print(f"train corpus creation creation: {_corpusTime:.3f}")
        self.model = ldamulticore.LdaMulticore(self.trainCorpus, 
                                               id2word=id2word,
                                               num_topics=self.nClasses, workers=self.nJobs, chunksize=self.batchSize, passes=self.maxIter)
        
        _trainTime = default_timer() - start - _dictTime - _corpusTime
        if verbose: print(f"model creation and training time: {_trainTime:.3f}")
        return self
    
    def predict(self, x, y=None, verbose=False):
        # needs a sparse array

        corpus = matutils.Sparse2Corpus(x.T)
        output = self.model.get_document_topics(corpus)
        _out = matutils.corpus2dense(output, num_terms=self.nClasses)

        predT = util.getTopPrediction(_out.T)

        return predT


    def perplexity(self, x, y=None, vebose=False):
        _corpus = matutils.Sparse2Corpus(x.T)
        _perp = np.exp(-1. * self.model.log_perplexity(_corpus))
        return _perp
    
    def coherence(self, x, nTop=10, y=None, verbose=False, vocab=None):
        corpus = matutils.Sparse2Corpus(x.T)
        cm = CoherenceModel(model=self.model, corpus=corpus, topn=nTop, coherence='u_mass')
        return cm.get_coherence()


    def print_topics(self, nTopics=None, nTopWords=10, verbose=True):
        # https://stackoverflow.com/questions/46536132/how-to-access-topic-words-only-in-gensim
        # https://stackoverflow.com/questions/15016025/how-to-print-the-lda-topics-models-from-gensim-python
        if (nTopics is None):
            nTopics = self.nClasses
        if (nTopics==0):
            return
        topicWords = {}
        filters = [lambda x: x.lower(), strip_punctuation, strip_numeric]
        for topic in self.model.show_topics(num_topics=nTopics, num_words=nTopWords, formatted=True):
            topicwords = preprocess_string(topic[1], filters)
            topicWords[topic[0]] = topicwords
        if verbose:
            for topic, words in topicWords.items():
                print('Topic: %d' % topic)
                print('  %s' % ', '.join(words))
        return topicWords


##################################################################################################################################################################################


### Below models are deprecated, but kept b/c we might want to update in the future.


### -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ###

class NMF(ClusteringModel):

    def __init__(self):
        super().__init__()

    def train(self, x, nClasses, maxIter=1000, y=None, verbose=False):
        self.model = nmf(n_components=nClasses, max_iter=maxIter, solver="mu", init="nndsvd", beta_loss="kullback-leibler", alpha_W=0.00005, alpha_H=0.00005, l1_ratio=1)
        output = self.model.fit_transform(x)
        pred = util.getTopPrediction(output)
        _silhouette, _calinskiHarabasz, _daviesBouldin, _homogeneity, _completeness, _vMeasure, _rand = util.getClusterMetrics(pred, x=x, labels=y, supervised=y is not None, verbose=verbose)
        return pred, [_silhouette, _calinskiHarabasz, _daviesBouldin, _homogeneity, _completeness, _vMeasure, _rand]
    

### -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ###

class KMeans(ClusteringModel):

    def __init__(self):
        super().__init__()

    def train(self, x, nClasses, batchSize=4096, maxIter=5000, y=None, verbose=False):
        self.model = km(n_clusters=nClasses, batch_size=batchSize, max_iter=maxIter)
        self.model.fit(x)
        pred = self.model.labels_
        _silhouette, _calinskiHarabasz, _daviesBouldin, _homogeneity, _completeness, _vMeasure, _rand = util.getClusterMetrics(pred, x=x, labels=y, supervised=y is not None, verbose=verbose)
        return pred, [_silhouette, _calinskiHarabasz, _daviesBouldin, _homogeneity, _completeness, _vMeasure, _rand]
