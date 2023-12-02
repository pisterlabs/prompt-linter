import pandas as pd
import pandas as pd
import numpy as np
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing import PorterStemmer
from gensim.parsing.preprocessing import STOPWORDS
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
from gensim.similarities.docsim import Similarity
import gensim.similarities.docsim as ds
from gensim.utils import SaveLoad
from nltk.stem import WordNetLemmatizer
import nltk
from dask import delayed
import json
import mysql_config
import pymysql
import datetime
import os
from io import BytesIO
import pickle
import boto3
import logging
logger = gensim.similarities.docsim.logger
class BufferShard(gensim.similarities.docsim.Shard):
    def __init__(self, fname, index):
            self.dirname, self.fname = os.path.split(fname)
            self.length = len(index)
            self.cls = index.__class__
            logger.info("saving index shard to %s", self.fullname())
            pickle_save(index, self.fullname())
            self.index = self.get_index()

    def get_index(self):
        if not hasattr(self, 'index'):
            logger.debug("mmaping index from %s", self.fullname())
            self.index = load_unpickle(self.fullname())
        return self.index
gensim.similarities.docsim.Shard = BufferShard

def pickle_save(obj, fname):
    pickled = pickle.dumps(obj)
    # stream = BytesIO(pickled)
    # s3.upload_fileobj(stream, bucket_name, fname)
    with open(fname, 'wb') as file:
        file.write(pickled)

def load_unpickle(fname):
    # stream = BytesIO()
    # s3.download_fileobj(bucket_name, fname, stream)
    with open(fname, 'rb') as file:
        obj = pickle.loads(file.read())
    # obj = pickle.loads(stream.getvalue())
    return obj

stemmer = PorterStemmer()
# s3 = boto3.client('s3')
# bucket_name = 'arxiv-models'

class Models:
    def __init__(self):
        attrs = ['idx_to_arxiv','tfidf_model','corpus_dict', 'similarity_index']
        for attr in attrs:
            try:
                self.__setattr__(attr, load_unpickle(attr + '.pckl'))
            except FileNotFoundError:
                self.__setattr__(attr, None)

    def __getattribute__(self, name):
        attr = super().__getattribute__(name)
        if attr is None:
            try:
                attr = load_unpickle(name + '.pckl')
                self.__setattr__(name, attr)
            except:
                logging.error(name + '.pckl could not be found')
                pass
        return attr

models = Models()


def lemmatize_stemming(text):
    try:
        return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))
    except LookupError:
        nltk.download('wordnet')
        return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))


def connect():
    return pymysql.connect(mysql_config.host,
                       user=mysql_config.name,
                       passwd=mysql_config.password,
                       connect_timeout=5,
                       database='arxiv',
                       port = mysql_config.port)
@delayed
def preprocess(text):
    result=[]
    for token in simple_preprocess(text) :
        if token not in STOPWORDS and len(token) > 2:
            result.append(lemmatize_stemming(token))
    return result


def get_tfidf(articles, tfidf_model = None, corpus_dict = None):
    articles_preprocessed = []
    for art in articles:
        articles_preprocessed.append(preprocess(art))

    # Evaluate dask delayed functions
    for i, art in enumerate(articles_preprocessed):
        articles_preprocessed[i] = art.compute()

    if corpus_dict is None:
        corpus_dict = Dictionary(articles_preprocessed)
        pickle_save(corpus_dict, 'corpus_dict.pckl')

    bow_corpus = [corpus_dict.doc2bow(doc) for doc in articles_preprocessed]
    if tfidf_model is None:
        print('Fitting tfidf model')
        tfidf_model = gensim.models.TfidfModel(bow_corpus, id2word=corpus_dict.id2token,)
        pickle_save(tfidf_model, 'tfidf_model.pckl')
    tfidf_corpus = [tfidf_model[doc] for doc in bow_corpus]
    return tfidf_corpus, corpus_dict

def create_index():
    conn = connect()
    df = pd.read_sql(""" SELECT id, title, summary FROM articles""", conn)

    articles = (df['title'] + '. ' + df['summary']).tolist()

    tfidf_corpus, corpus_dict = get_tfidf(articles)

    index = Similarity('index', tfidf_corpus, num_features=len(corpus_dict))
    pickle_save(index, 'similarity_index.pckl')
    pickle_save(df['id'].to_dict(), 'idx_to_arxiv.pckl')
    conn.close()


def get_recommendations(user_id, cutoff_days = 20, no_papers=10,
                        based_on = None):
    conn = connect()
    if based_on is None:
        df_bookmarks = pd.read_sql(""" SELECT
                                       articles.id as id,
                                       bookmarks.user_id as user_id,
                                       DATE(updated) as updated,
                                       authors,
                                       title,
                                       summary
                                       FROM articles
                                       INNER JOIN bookmarks
                                       ON articles.id = bookmarks.article_id
                                       WHERE bookmarks.user_id = {}
                                       AND DATE(updated) > DATE_ADD(DATE(NOW()), INTERVAL {:d} day)""".format(user_id, -cutoff_days), conn)
    else:
        df_bookmarks = pd.DataFrame(based_on)
    if len(df_bookmarks):
        try:
            articles = (df_bookmarks['title'] + '. ' + df_bookmarks['summary']).tolist()
            tfidf, _ = get_tfidf(articles, models.tfidf_model, models.corpus_dict)

            sim = models.similarity_index[tfidf]

            no_bm = len(df_bookmarks)
            sim = np.argsort(sim, axis=-1)[:,::-1].T.flatten()[:(no_papers)*(no_papers)]
            _, unq = np.unique(sim, return_index=True)
            sim = sim[np.sort(unq)]
            sim = sim[:no_papers+no_bm]
            rec_id = {models.idx_to_arxiv[s]:i for i, s in enumerate(sim[no_bm:])}

            rec = pd.read_sql(""" SELECT * from articles
                         WHERE id in ('{}')
                         ORDER BY updated DESC""".format("','".join(rec_id.keys())), conn)
            rec['updated'] = rec['updated'].apply(str)
            ordering = [rec_id[id] for id in rec['id']]
            sim[no_bm:] = [sim[no_bm:][idx] for idx in ordering]
            A = np.zeros([len(sim),len(sim)])
            for i, s in enumerate(sim):
                A[i,:] =  models.similarity_index.similarity_by_id(s)[sim]

            df_bookmarks['updated'] = df_bookmarks['updated'].apply(str)
            conn.close()
            return rec, A, df_bookmarks
        except Exception as e:
            conn.close()
            df_bookmarks['updated'] = df_bookmarks['updated'].apply(str)
            return pd.DataFrame(), np.ones([len(df_bookmarks),len(df_bookmarks)]), df_bookmarks
    else:
        conn.close()
        df_bookmarks['updated'] = df_bookmarks['updated'].apply(str)
        return pd.DataFrame(), np.ones([len(df_bookmarks),len(df_bookmarks)]), df_bookmarks


if __name__ == '__main__':
    create_index()
