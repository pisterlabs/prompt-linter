import tomotopy as tp
import gensim.corpora as corpora
from gensim.models import CoherenceModel
import numpy as np
import pyLDAvis


class LDAModel:
    def __init__(self, tw, min_cf, rm_top, k, eta, alpha, seed, model_name):
        self.tw = tw
        self.min_cf = min_cf
        self.rm_top = rm_top
        self.k = k
        self.eta = eta
        self.alpha = alpha
        self.seed = seed
        self.model_name = model_name.lower()
        if self.model_name == 'lda':
            self.lda = tp.LDAModel(
                tw=self.tw,
                min_cf=self.min_cf,
                rm_top=self.rm_top,
                k=self.k,
                eta=self.eta,
                alpha=self.alpha,
                seed=self.seed,
            )
        elif self.model_name == 'hdp':
            self.lda = tp.HDPModel(
                tw=self.tw,
                min_cf=self.min_cf,
                rm_top=self.rm_top,
                eta=self.eta,
                alpha=self.alpha,
                seed=self.seed,
            )

        elif self.model_name == 'pa':
            self.lda = tp.PAModel(
                tw=self.tw,
                min_cf=self.min_cf,
                rm_top=self.rm_top,
                eta=self.eta,
                alpha=self.alpha,
                seed=self.seed,
            )
        elif self.model_name == 'drm':
            self.lda = tp.DMRModel(
                tw=self.tw,
                min_cf=self.min_cf,
                rm_top=self.rm_top,
                eta=self.eta,
                alpha=self.alpha,
                seed=self.seed,
            )
        elif self.model_name == 'hdp':
            self.lda = tp.HDPModel(
                tw=self.tw,
                min_cf=self.min_cf,
                rm_top=self.rm_top,
                eta=self.eta,
                alpha=self.alpha,
                seed=self.seed,
            )
        elif self.model_name == 'llda':
            self.lda = tp.LLDAModel(
                tw=self.tw,
                min_cf=self.min_cf,
                rm_top=self.rm_top,
                k=self.k,
                eta=self.eta,
                alpha=self.alpha,
                seed=self.seed,
            )
        else:
            raise("This model is not handled")
        self.text = None

    def fit(self, text):
        self.text = text
        for vec in self.text:
            if vec:
                self.lda.add_doc(vec)
        self.lda.burn_in = 100

    def train(self, iterations):
        for i in range(0, iterations):
            self.lda.train(1)
            print(
                "Iteration: {}\tLog-likelihood: {}\tNum. of topics: {}".format(
                    i, self.lda.ll_per_word, self.k
                )
            )

    def get_topics(self, top_n):
        sorted_topics = [
            k
            for k, v in sorted(
                enumerate(self.lda.get_count_by_topics()),
                key=lambda x: x[1],
                reverse=True,
            )
        ]
        topics = dict()
        for k in sorted_topics:
            topic_wp = []
            for word, prob in self.lda.get_topic_words(k, top_n=top_n):
                topic_wp.append((word, prob))
            topics[k] = topic_wp
        return topics

    def get_coherence(self):
        vocab = corpora.Dictionary(self.text)
        corpus = [vocab.doc2bow(words) for words in self.text]
        topic_list = []
        for k, tups in self.get_topics(100).items():
            topic_tokens = []
            for w, p in tups:
                topic_tokens.append(w)
            topic_list.append(topic_tokens)
        cm = CoherenceModel(
            topics=topic_list,
            corpus=corpus,
            dictionary=vocab,
            texts=self.text,
            coherence="c_v",
        )
        return cm.get_coherence()

    def get_pyLDAvis(self, topic_n=10):
        from IPython.display import HTML

        topic_term_dists = np.stack(
            [self.lda.get_topic_word_dist(k) for k in range(topic_n)]
        )
        doc_topic_dists = np.stack([doc.get_topic_dist() for doc in self.lda.docs])
        doc_lengths = np.array([len(doc.words) for doc in self.lda.docs])
        vocab = list(self.lda.used_vocabs)
        term_frequency = self.lda.used_vocab_freq

        prepared_data = pyLDAvis.prepare(
            topic_term_dists, doc_topic_dists, doc_lengths, vocab, term_frequency
        )
        return prepared_data
    
    def get_drm_topics(self, top_n):
        """
        Returns a dictionary of topics and their associated word-probability pairs.
        """
        sorted_topics = sorted(enumerate(self.lda.get_count_by_topics()), key=lambda x: x[1], reverse=True)
        topics = {k: self.lda.get_topic_words(k, top_n) for k, v in sorted_topics}
        return topics
