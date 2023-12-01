from pathlib import Path
from typing import Dict

import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
from textblob import TextBlob

from functions.textprocessing import TextProcessor


# A class to enable natural language processing based analysis
# Has methods for collecting tweet sentiment and the creation and use of the TF-IDF similarity model
class NLPAnalyser:
    # Create the id2word, TF-IDF, and similarity model
    def __init__(self):
        self.id2word = None
        self.sim_model = None
        self.tf_idf = None
        pass

    # Performs textual NLP analysis and returns the sentiment of a Tweet
    @staticmethod
    def get_tweet_sentiment(tweet: str) -> str:
        analysis = TextBlob(TextProcessor.clean_tweet(tweet))
        if analysis.sentiment.polarity > 0:
            return 'positive'
        elif analysis.sentiment.polarity == 0:
            return 'neutral'
        else:
            return 'negative'

    # Method that creates the topic model from a list of documents
    # Assumed that the documents have not been cleaned - will be cleaned as a result
    def create_tfidf_model(self, scraped_data: Dict):
        cleaned_documents = [scraped_data[k].cleaned_tokens for k in scraped_data.keys() if scraped_data[k] is not None]
        self.id2word = corpora.Dictionary(cleaned_documents)
        corpus = [self.id2word.doc2bow(text) for text in cleaned_documents]
        self.tf_idf = gensim.models.TfidfModel(corpus)
        self.sim_model = gensim.similarities.SparseMatrixSimilarity(self.tf_idf[corpus],
                                                                    num_features=len(self.id2word))

    # Method to check the similarity of a given document to the TF-IDF similarity model created
    def check_similarity(self, document):
        if document is None:
            return 0
        # collect the cleaned tokens from the document
        cleaned_tokens = [document.cleaned_tokens]

        # create a bag of words with the cleaned tokens and collect TF-IDF representation
        test_corpus = [self.id2word.doc2bow(cleaned_tokens[0])]
        query_test_words = self.tf_idf[test_corpus]

        # check the maximum similarity for the document
        for doc in query_test_words:
            max_sim = max(self.sim_model[doc])
        return max_sim
