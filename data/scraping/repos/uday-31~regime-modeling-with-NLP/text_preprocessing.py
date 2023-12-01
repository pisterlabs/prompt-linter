import gensim
from gensim.corpora import Dictionary
from gensim.matutils import corpus2dense, corpus2csc
from gensim.models import TfidfModel
from gensim.models.nmf import Nmf
from gensim.models.coherencemodel import CoherenceModel
import numpy as np
import pandas as pd
import nltk

nltk.download("stopwords")
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

FOMC_STOP_WORDS = ["federal", "reserve", "board", "meeting", "committee", "minutes", "members"]


def remove_names_from_minutes(text: str):
    """
    This function removes all names from the start of FED Minutes by relying on
    the fact that the phrases 'the manager' and 'unanimous' tend to appear at
    the end of the initial string of names.

    @param tet(str): text which needs to have names removed from the start
    @returns res(str): portion of text after first occurence of 'the manager'
                       or 'unanimous'
    """
    text = text.lower()
    split_by = ''
    if 'the manager' in text and 'unanimous' in text:
        if text.index('the manager') > text.index('unanimous'):
            split_by = 'unanimous'
        else:
            split_by = 'the manager'
    elif 'the manager' in text:
        split_by = 'the manager'
    elif 'unanimous' in text:
        split_by = 'unanimous'
    else:
        raise ValueError('Neither in text!')
    
    res = text.split(split_by)[1]
    return res


def tokenizer_wo_stopwords(text: str):
    """
    This function prepares raw text by tokenizing it and removing all stop
    words (based on nltk stopwords).

    @param text(str): raw text which needs to be prepared for analysis
    @return res(str): string representation of text without stopwords
    """
    tokens = nltk.word_tokenize(text)
    words = [word.lower() for word in tokens if word.isalpha()]
    words_wo_stop = [w.lower() for w in words if
                     w.lower() not in ENGLISH_STOP_WORDS and w.lower() not in FOMC_STOP_WORDS]
    res = ' '.join(words_wo_stop)
    return res


class TF_IDF():
    def __init__(self, X_train: pd.Series = None, X_test: pd.Series = None):
        self.X_train = X_train
        self.X_test = X_test
        
        # Attributes needed for manual TF-IDF computations
        self.def_vectorizer = None
        self.tfidf_manual_train = None
        self.tfidf_manual_test = None
        
        # Attributes needed for gensim TF-IDF computations
        self.dict_gensim_statements = None
        self.tfidf_model_gensim = None
        self.tfidf_statements_train = None
        self.tfidf_statements_test = None
        self.tfidf_gensim_train = None
        self.tfidf_gensim_test = None
    
    def assign(self, X_train: pd.Series = None, X_test: pd.Series = None):
        self.X_train = X_train
        self.X_test = X_test
    
    def fit_manual_helper(self, train: bool = True):
        """
        This function manually computes the TF-IDF values for a column of train
        OR test documents, to avoid the incorrect computations performed by
        sklearn's native implementation.
        
        @param train: flag determining if function will fit/transform train
                      data, or only fit vectorizer to test data
        """
        if train:
            text = self.X_train
        else:
            text = self.X_test
        
        try:
            assert text is not None
        except Exception as e:
            print(f"assign() train/test data before fitting!")
            return
        
        # Get number of documents
        n_docs = text.shape[0]
        
        # Generate bag-of-words matrix
        if train:
            self.def_vectorizer = CountVectorizer(token_pattern='[a-zA-Z]+')
            word_bow_matrix = self.def_vectorizer.fit_transform(text)
        else:
            word_bow_matrix = self.def_vectorizer.transform(text)
        
        word_bow_df = pd.DataFrame(
            word_bow_matrix.toarray(),
            columns=self.def_vectorizer.get_feature_names_out()
        )
        
        # Create TF matrix
        tf_df = word_bow_df / word_bow_df.sum(axis=1).values.reshape(n_docs, 1)
        
        # Compute IDF values
        idf = np.log(n_docs / (word_bow_df / word_bow_df.values).sum(axis=0))
        
        # Manually create TF-IDF matrix
        if train:
            self.tfidf_manual_train = tf_df * idf
        else:
            self.tfidf_manual_test = tf_df * idf
    
    def fit_manual(self):
        """
        This function fits the manual TF-IDF model to train data and generates
        the values for the test data by calling the previously-defined helper
        function consecutively on train and test data.
        """
        self.fit_manual_helper(train=True)
        self.fit_manual_helper(train=False)
    
    def fit_gensim_helper(self, train: bool = True):
        """
        This function uses gensim to compute the TF-IDF values for a column of
        train or test documents, to avoid the incorrect computations performed
        by sklearn's native implementation.
    
        @param train: flag determining if function will fit/transform train
                      data, or only fit vectorizer to test data
        """
        if train:
            text = self.X_train
        else:
            text = self.X_test
        
        try:
            assert text is not None
        except Exception as e:
            print(f"assign() train/test data before fitting!")
            return
        
        gensim_statements = text.apply(lambda x: x.split(" ")).tolist()
        
        if train:
            self.dict_gensim_statements = Dictionary(gensim_statements)
        
        bow_gensim_statements = [self.dict_gensim_statements.doc2bow(d) for d in gensim_statements]
        
        if train:
            self.tfidf_model_gensim = TfidfModel(bow_gensim_statements)
        
        tfidf_statements = self.tfidf_model_gensim[bow_gensim_statements]
        
        if train:
            self.tfidf_statements_train = tfidf_statements
        else:
            self.tfidf_statements_test = tfidf_statements
        
        num_terms = len(self.dict_gensim_statements.keys())
        num_docs = len(tfidf_statements)
        
        if train:
            self.tfidf_gensim_train = corpus2dense(
                tfidf_statements,
                num_terms,
                num_docs
            ).T
        else:
            self.tfidf_gensim_test = corpus2dense(
                tfidf_statements,
                num_terms,
                num_docs
            ).T
    
    def fit_gensim(self):
        """
        This function fits the gensim TF-IDF model to train data and generates
        the values for the test data by calling the previously-defined helper
        function consecutively on train and test data.
        """
        self.fit_gensim_helper(train=True)
        self.fit_gensim_helper(train=False)


if __name__ == "__main__":
    print(f"Please import this module as a library!")
