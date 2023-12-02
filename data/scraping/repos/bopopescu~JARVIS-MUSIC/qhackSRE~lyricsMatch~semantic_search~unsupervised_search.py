import pandas as pd
import numpy as np
from collections import Counter
import re
from lyricsMatch.semantic_search.search_init import data, count_vect_lyric, tf_transformer_lyric, words_tfidf_lyric, count_vect_name, tf_transformer_name, words_tfidf_name, model

# languange processing imports
#import nltk
import string
from gensim.corpora import Dictionary
# preprocessing imports
from sklearn.preprocessing import LabelEncoder
from pprint import pprint
# model imports
from gensim.utils import simple_preprocess

from gensim.models import CoherenceModel, Phrases, phrases
from gensim.models.doc2vec import LabeledSentence
from gensim.models.doc2vec import TaggedDocument
from gensim.models.doc2vec import Doc2Vec
from gensim.test.utils import get_tmpfile
from sklearn.preprocessing import normalize

#import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import scipy
import warnings
warnings.filterwarnings("ignore",category=FutureWarning)
warnings.filterwarnings("ignore",category=DeprecationWarning)


def process_query(count_vect, tf_transformer, lyric):
    #lyric_no_stops = remove_stopwords(lyric)
    #lyric_lemmatized = lemmatization(lyric_no_stops, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    ##lyric_lemmatized = lemmatize_more(lyric_lemmatized)
    #lyric = list(map(lambda words: " ".join(words), lyric_lemmatized))
    words_count = count_vect.transform(lyric)
    search_tfidf = tf_transformer.transform(words_count)
    return search_tfidf

def search_top5(words_tfidf, search_tfidf, data):
    matrix_dot = np.dot(words_tfidf, search_tfidf[0].transpose())
    #print(type(matrix_dot.todense()))
    matrix_dot = matrix_dot.todense()
    #sorted_index = matrix_dot.argsort(axis=0)
    #sorted_index = np.array(sorted_index)[:-6:-1]
    return matrix_dot
    #for iindex in sorted_index:
    #    print(data.iloc[iindex].song)
    #print(sorted_index)

def search_lyrics(lyric):
    lyric = [lyric]
    # lyric = ['Overwhelming Everything about you is so overwhelming']
    # lyric = ['I think I got one Her soul is presidential like Barack']
    # lyric = ['To the left  to the left To the left  to the left']
    search_tfidf_lyric = process_query(count_vect_lyric, tf_transformer_lyric, lyric)
    matrix_lyric = search_top5(words_tfidf_lyric, search_tfidf_lyric, data)

    search_tfidf_name = process_query(count_vect_name, tf_transformer_name, lyric)
    matrix_name = search_top5(words_tfidf_name, search_tfidf_name, data)

    inferred_vector = model.infer_vector(lyric[0].split(' '))
    sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
    rank = [[data.index.get_loc(int(docid.strip('doc'))), sim] for docid, sim in sims]

    rank_df = pd.DataFrame(rank, columns=['docid', 'sim'])
    rank_df = rank_df.sort_values(by=['docid'])
    rank_df.index = rank_df.docid
    rank_df = rank_df.drop(columns=['docid'])

    matrix_d2v = rank_df.as_matrix()

    # output is in a dict. 5 items. key varies but always different. each item is a list. list[0] sone name, list[1] singer name, list[2] youtube link
    matrix_dot = 0.7 * matrix_lyric + 0.1 * matrix_name + 0.2 * matrix_d2v
    sorted_index = matrix_dot.argsort(axis=0)
    sorted_index = np.array(sorted_index)[:-6:-1]
    result_dict = dict()
    link = 'https://www.youtube.com/results?search_query='
    for iindex in sorted_index:
        parameter = data.iloc[iindex].song.values[0].replace(' ', '+') + '+' + data.iloc[iindex].singer.values[
            0].replace(' ', '+')
        #print(link + parameter.lower())
        result = []
        result.append(data.iloc[iindex].song.values[0])
        result.append(data.iloc[iindex].singer.values[0])
        result.append(link + parameter.lower())
        result_dict[iindex[0]] = result
    return result_dict
        #    print(data.iloc[iindex].song)

if  __name__ == '__main__':
    print(search_lyrics('Overwhelming Everything about you is so overwhelming'))
    print(search_lyrics('Overwhelming Everything about you is so overwhelming'))
