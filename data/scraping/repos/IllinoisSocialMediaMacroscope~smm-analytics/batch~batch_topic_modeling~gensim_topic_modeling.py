from nltk.corpus import stopwords
import re
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from nltk import WordNetLemmatizer
import pyLDAvis
import pyLDAvis.gensim


class Gensim_Topic_Modeling:

    def __init__(self, df, column):
        self.data = df[df[column] != ''][column].dropna().astype(
            'str').tolist()

    def preprocessing(self):
        self.data = [re.sub('\S*@\S*\s?', "", sent) for sent in self.data]
        self.data = [re.sub('\s+', ' ', sent) for sent in self.data]
        self.data = [re.sub("\'", "", sent) for sent in self.data]
        self.data = [re.sub(r"http\S+", "", sent) for sent in self.data]
        self.data = [re.sub(r"rt\S+", "", sent) for sent in self.data]

        data_words = list(self.sent_to_words(self.data))

        # Build the bigram
        bigram = gensim.models.Phrases(data_words, min_count=5, threshold=50)
        # higher threshold fewer phrases.

        # Remove Stop Words
        data_words_nostops = self.remove_stopwords(data_words)

        # Form Bigrams
        bigram_mod = gensim.models.phrases.Phraser(bigram)
        data_words_bigrams = self.make_bigrams(bigram_mod, data_words_nostops)

        # lemmatization keeping only noun, adj, vb, adv
        data_lemmatized = self.lemmatization(data_words_bigrams)

        # Create Dictionary
        id2word = corpora.Dictionary(data_lemmatized)

        # Term Document Frequency
        corpus = [id2word.doc2bow(text) for text in data_lemmatized]

        return data_lemmatized, id2word, corpus

    def build_lda_model(self, num_topics, corpus, id2word):
        lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                    id2word=id2word,
                                                    num_topics=num_topics,
                                                    random_state=100,
                                                    update_every=1,
                                                    chunksize=100,
                                                    passes=10,
                                                    alpha='auto',
                                                    per_word_topics=True)

        return lda_model

    def lda_model_metrics(self, lda_model, corpus, id2word, data_lemmatized):
        perplexity = lda_model.log_perplexity(corpus)
        coherence_model_lda = CoherenceModel(model=lda_model,
                                             texts=data_lemmatized,
                                             dictionary=id2word,
                                             coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        metrics = [["perplexity", "coherence score"],
                   [perplexity, coherence_lda]]

        return metrics

    def visualize_lda_model(self, lda_model, corpus, id2word):
        data = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
        html = pyLDAvis.prepared_data_to_html(data)

        return html

    @staticmethod
    def sent_to_words(sentences):
        for sentence in sentences:
            yield (gensim.utils.simple_preprocess(str(sentence), deacc=True))
            # deacc=True removes punctuations

    @staticmethod
    def remove_stopwords(texts):
        stop_words = stopwords.words('english')
        return [[word for word in simple_preprocess(str(doc)) if
                 word not in stop_words] for doc in texts]

    @staticmethod
    def make_bigrams(bigram_mod, texts):
        return [bigram_mod[doc] for doc in texts]

    @staticmethod
    def lemmatization(texts):
        texts_out = []
        wnl = WordNetLemmatizer()
        for sent in texts:
            texts_out.append([wnl.lemmatize(t) for t in sent])

        return texts_out
