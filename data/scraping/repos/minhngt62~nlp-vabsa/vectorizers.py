from sklearn.feature_extraction.text import TfidfVectorizer
import gensim
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
import os
import numpy as np

class DenseTfidfVectorizer(TfidfVectorizer):
    def transform(self, raw_documents):
        X = super().transform(raw_documents)
        return X.toarray()

    def fit_transform(self, raw_documents, y=None):
        X = super().fit_transform(raw_documents, y=y)
        return X.toarray()
    
class LDA:
    def __init__(
        self, 
        ckpts_dir="checkpoints", 
        num_topics=48, 
        alpha=0.1, 
        eta=0.1, 
        iterations=1000
    ):
        self.ckpts_dir = ckpts_dir
        self.params = {'num_topics':num_topics, 'alpha':alpha, 'eta':eta, 'iterations':iterations}
        self.model = None

    def fit(self, X, y=None):
        self.X = X
        self.dictionary = Dictionary(text.split() for text in X)
        self.train_corpus = [self.dictionary.doc2bow(text.split()) for text in X] # (word_idx, freq_count)
        self.model = gensim.models.LdaMulticore(id2word=self.dictionary, minimum_probability=0.000, **self.params)
        self.model.update(self.train_corpus)
        return self

    def save(self, model_name='test'):
        os.makedirs(os.path.join(self.ckpts_dir, model_name), exist_ok=True)
        self.model.save(os.path.join(self.ckpts_dir, model_name, model_name))
    
    def load(self, model_name='test'):
        self.model = gensim.models.LdaMulticore.load(os.path.join(self.ckpts_dir, model_name, model_name))
        self.dictionary = self.model.id2word

    def predict(self, document:str):
        document = document.split()
        document = self.dictionary.doc2bow(document)
        topics =  self.model.get_document_topics(document)
        result = []
        for topic in topics:
            result.append(topic[1])
        return np.array(result)
    
    def score(self, *args, **kwargs):
        score_fn = CoherenceModel(model=self.model, texts=[text.split() for text in self.X], dictionary=self.dictionary, coherence='c_v')
        return score_fn.get_coherence()
    
    def get_params(self, deep=False):
        return self.params
    
    def set_params(self, **parameters):
        self.params = parameters
        return self