from src.analyzer.engines.engine import ModelingEngine

from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
from gensim.models.tfidfmodel import TfidfModel
from gensim.models import Nmf

class GensimEngine(ModelingEngine):
    """
    GensimEngine implements the methods for fitting and evaluating gensim models on a
    corpus.
    """
    def __init__(self, engine="lda", dictionary=None, min_topics=5, max_topics=10, **kwargs):
        if self.engine not in ["lda", "nmf"]:
            raise ValueError("engine must be either 'lda' or 'nmf'.")
        self.engine = engine
        self.dictionary = dictionary
        self.engine_opts = kwargs
        self.min_topics = min_topics
        self.max_topics = max_topics
        self.models_ = {}
        self.corpus_ = None

    def fit(self, corpus):
        """
        Fits a set of models on the provided corpus.
        """
        if self.engine == "lda":
            self._fit_models(LdaModel, corpus, **self.engine_opts)
        elif self.engine == "nmf":
            tfidf = TfidfModel(dictionary=self.dictionary)
            self._fit_models(Nmf, tfidf[corpus], **self.engine_opts)
        else:
            raise ValueError("engine must be either lda or nmf.")
        self.corpus_ = corpus

    def _fit_models(self, model_class, corpus, **kwargs):
        """
        Fits several models with different numbers of topics on the given corpus.
        """
        for num_topics in range(self.min_topics, self.max_topics + 1):
            print("fitting {} model with {} topics".format(model_class.__name__, num_topics))
            model = model_class(corpus=corpus, id2word=self.dictionary, num_topics=num_topics, **kwargs)
            cm = CoherenceModel(model=model, corpus=corpus, dictionary=self.dictionary, coherence='u_mass')
            results = {}
            results['coherence'] = cm.get_coherence()
            print("coherence: {}".format(results['coherence']))
            self.models_[num_topics] = {}
            self.models_[num_topics]['model'] = model
            self.models_[num_topics]['results'] = results

    def update(self, documents):
        """
        Updates the set of models with a stream of new corpus documents.
        """
        bow = [self.dictionary.doc2bow(doc, allow_update=True) for doc in documents]
        self.corpus_.append(bow)
        for _, m in self.models_.items():
            m['model'].update(bow)
            cm = CoherenceModel(model=m['model'], corpus=self.corpus_, dictionary=self.dictionary, coherence='u_mass')
            m['results']['coherence'] = cm.get_coherence()

    def topics(self):
        """
        Return the discovered topics and evaluation metrics for the set of current models.
        """
        topics = {}
        for k, m in self.models_.items():
            topics[k] = {}
            topics[k]['coherence'] = m['results']['coherence']
            topics[k]['topics'] = m['model'].show_topics(num_topics=k)
        return topics