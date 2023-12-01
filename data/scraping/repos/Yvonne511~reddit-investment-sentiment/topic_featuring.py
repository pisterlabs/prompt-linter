import pandas as pd
import pathlib
import os
import matplotlib.pyplot as plt

import nltk
import ssl
import gensim
import gensim.corpora as corpora
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel

from pprint import pprint

import spacy

import pickle
import re
import pyLDAvis
import pyLDAvis.gensim

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download('stopwords')
from nltk.corpus import stopwords

class topic_featuring:
    texts = []
    df_submissions = pd.DataFrame()
    tokens = []
    id2word = None
    corpus = None
    lda_model = None

    def getTexts(self):
        cwd = os.getcwd()
        target_dir = os.path.join(cwd, 'data-submission')
        file_path_array = []
        # Get all the files in dir
        for filepath in pathlib.Path(target_dir).glob('**/*'):
            if filepath.is_file():
                file_path_array.append(filepath.relative_to(cwd).as_posix())
        for file_path in file_path_array:
            file_path = os.path.join(cwd, file_path)
            df = pd.read_csv(file_path)
            self.df_submissions = pd.concat([self.df_submissions, df], ignore_index=True)
        self.df_submissions = self.df_submissions.drop_duplicates(keep='first')
        self.df_submissions = self.df_submissions.drop(columns=['upvote_ratio'])
        ## merge body and title into one column and remove emoty or removed body
        self.df_submissions['body'] = self.df_submissions['body'].fillna('').apply(lambda x: 'deletedpost' if x == '[removed]' else x)
        self.df_submissions['text'] = self.df_submissions['title'] + ' ' + self.df_submissions['body']
        self.texts = self.df_submissions['text'].values.tolist()
    
    def cleanText(self):
        # Lowercase
        self.texts = [text.lower() for text in self.texts]
        # Remove urls
        self.texts = [re.sub(r'http\S+', '', text) for text in self.texts]
        # Remove characters
        self.texts = [re.sub(r'[^a-zA-Z0-9\s]', '', text) for text in self.texts]
        # Remove new lines
        self.texts = [re.sub(r'\n', ' ', text) for text in self.texts]
        # Remove stop words
        stop_words = stopwords.words('english')
        self.tokens = [[word for word in text.split() if word not in stop_words] for text in self.texts]
    
    def LDAModel(self):
        self.id2word = corpora.Dictionary(self.tokens)
        self.corpus = [self.id2word.doc2bow(text) for text in self.tokens]
        self.lda_model = gensim.models.ldamodel.LdaModel(corpus=self.corpus,
                                                id2word=self.id2word,
                                                num_topics=4)

    def visualization(self):
        lda_model = self.lda_model
        vis = pyLDAvis.gensim.prepare(lda_model, self.corpus, self.id2word)
        pyLDAvis.save_html(vis, 'lda.html')

tf = topic_featuring()
tf.getTexts()
tf.cleanText()
tf.LDAModel()
tf.visualization()


