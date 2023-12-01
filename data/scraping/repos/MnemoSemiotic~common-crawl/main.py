
import src.models as models
import src.pre_clean as clean
import src.wordcloud as wc
import src.plotting as pt
import pandas as pd
import numpy as np
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
import nltk
import os
import logging
import pyLDAvis.gensim
import json
import warnings
warnings.filterwarnings('ignore')  # To ignore all warnings that arise here to enhance clarity

from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
from gensim.corpora.dictionary import Dictionary
from numpy import array
import pyLDAvis.sklearn


if __name__ == '__main__':
    #
    # # Our data
    # filenames = ['crawl-data/corpus_csv.csv']
    #
    # filenames_clean = ['crawl-data/corpus_csv.txt']
    #
    # names = ['sites']
    # names_clean = ['sites (clean)']
    #
    # vectorizer = CountVectorizer(input='filename')
    # vectorizer_clean = CountVectorizer(input='filename')
    #
    # # get sparse matrix of file word counts
    # dtm = vectorizer.fit_transform(filenames)
    # dtm_clean = vectorizer_clean.fit_transform(filenames_clean)
    #
    # # build a list of the document vocabularies
    # #    - note, this vocab is built but contains
    # #      a lot of garbage
    # vocab = vectorizer.get_feature_names()
    # len(vocab) # 34152 total words for all, raw data
    # vocab[32000] # ultraviolet
    # # clean vocab
    # vocab_clean = vectorizer_clean.get_feature_names()
    # len(vocab_clean) # 29257 total words for data after cleaning
    #
    # # to get # of occurrences of a term, need to convert from
    # #    sparse matrix to numpy matrix
    # dtm_array = dtm.toarray()
    # # convert vocab to np.array
    # vocab = np.array(vocab)
    #
    # # Look at the number of occurrences of a few words in the raw data
    # #   - distribution
    # dtm_array[0, vocab=='distribution'] # appears 39 times biology
    # dtm_array[1, vocab=='distribution'] # appears 799 times in datascience
    # dtm_array[2, vocab=='distribution'] # appears 19 times in history
    # #   - cell
    # dtm_array[0, vocab=='cell'] # appears 1672 times biology
    # dtm_array[1, vocab=='cell'] # appears 62 times in datascience
    # dtm_array[2, vocab=='cell'] # appears 0 times in history
    # #   - war
    # dtm_array[0, vocab=='war'] # appears 0 times biology
    # dtm_array[1, vocab=='war'] # appears 0 times in datascience
    # dtm_array[2, vocab=='war'] # appears 1019 times in history
    # #   - cosine
    # dtm_array[0, vocab=='cosine'] # appears 0 times biology
    # dtm_array[1, vocab=='cosine'] # appears 8 times in datascience
    # dtm_array[2, vocab=='cosine'] # appears 0 times in history
    # #   - whether
    # dtm_array[0, vocab=='whether'] # appears 11 times biology
    # dtm_array[1, vocab=='whether'] # appears 175 times in datascience
    # dtm_array[2, vocab=='whether'] # appears 10 times in history
    #
    #
    #
    #
    #











    ''' For generating wordclouds on all data '''
    # datascience cards
    data = 'crawl-data/corpus.csv'
    df_corpus = pd.read_csv(data, sep='\t', names=['document'])
    df = df_corpus.reindex()

    df

    list_of_strings = [str(i) for i in df['document']]
    list_of_strings

    wc.create_wordcloud(' '.join(list_of_strings))


    df_corpus[15]
    # df_corpus_clean = clean.clean_dataframe(df_corpus)
    # df_corpus_clean[79]
    datascience_tfidf, datascience_count = models.get_vectorizers(df_datascience)
    # Export datascience cleaned
    np.savetxt(r'data/datascience_flashcards_cleaned.txt', df_datascience.values, fmt='%s')
    wc.create_wordcloud_from_df(df_corpus)


    #
    # # biology cards
    # data = 'data/biology_flashcards.txt'
    # df_biology = clean.read_cards(data)
    # df_biology_clean = clean.clean_dataframe(df_biology)
    # df_biology = clean.collapse_df(df_biology_clean)
    # df_biology[79]
    # np.savetxt(r'data/biology_flashcards_cleaned.txt', df_biology.values, fmt='%s')
    # biology_tfidf, biology_count = models.get_vectorizers(df_biology)
    # wc.create_wordcloud_from_df(df_biology)

    # amask = df_biology.str.contains('amask')
    # len(amask[amask==True])
    # amask[amask==True]
    # df_biology[7000]
    # df_collapsed[79]

    # history cards
    # data = 'data/history_flashcards.txt'
    # df_history = clean.read_cards(data)
    # df_history_clean = clean.clean_dataframe(df_history)
    # df_history = clean.collapse_df(df_history_clean)
    # df_history[79]
    # np.savetxt(r'data/history_flashcards_cleaned.txt', df_history.values, fmt='%s')
    # history_tfidf, history_count = models.get_vectorizers(df_history)
    # wc.create_wordcloud_from_df(df_history)



    ''' Topic Modeling with pyLDAvis'''
    # compile single corpora
    #   datascience cards
    data = 'data/datascience_flashcards.txt'
    df_datascience = clean.read_cards(data)
    #   biology cards
    data = 'data/biology_flashcards.txt'
    df_biology = clean.read_cards(data)
    #   history cards
    data = 'data/history_flashcards.txt'
    df_history = clean.read_cards(data)

    len(df_datascience)
    len(df_biology)
    len(df_history)

    frames = [df_datascience, df_biology, df_history]

    corpus = pd.concat(frames)
    # save out full file


    len(corpus) # 36188 rows


    # clean corpus
    corpus_clean = clean.clean_dataframe(corpus)
    corpus_collapsed = clean.collapse_df(corpus_clean)

    # save out full file
    np.savetxt(r'data/full_corpus_cleaned.txt', corpus_collapsed.values, fmt='%s')

    # Convert corpus to tf-idf and CountVectorizer
    #   - For pyLDAvis, need to keep the vectorizers
    tfidf_vectorizer = TfidfVectorizer(min_df=5, max_df=0.80, stop_words='english', lowercase=True, token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')

    count_vectorizer = CountVectorizer(min_df=5, max_df=0.80, stop_words='english', lowercase=True, token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')

    corpus_tfidf = tfidf_vectorizer.fit_transform(corpus_collapsed)
    corpus_count = count_vectorizer.fit_transform(corpus_collapsed)

    corpus_tfidf.shape # (36188, 9436)

    # Create wordmap from entire corpus
    # wc.create_wordcloud_from_df(corpus_collapsed)

    ''' Fit Latent Dirichlet Allocation Models '''
    # With Count Vector, running with default 10 iterations
    lda_corpus_count = LatentDirichletAllocation(n_topics=3, random_state=0)
    lda_corpus_count.fit(corpus_count)

    # With TF-IDF matrix, running with default 10 iterations
    lda_corpus_tfidf = LatentDirichletAllocation(n_topics=3, random_state=0)
    lda_corpus_tfidf.fit(corpus_tfidf)

    # not working in hydrogen
    # pyLDAvis.sklearn.prepare(lda_corpus_count, corpus_count, count_vectorizer)





    '''hey'''






















































    '''
    DATA CLEANING Dev
    '''
    data = 'data/datascience_flashcards.txt'
    df = clean.read_cards(data)
    df['question'][79]

    df_clean = clean.clean_dataframe(df)

    # df_clean.tail()

    df_collapsed = clean.collapse_df(df_clean)
    df['question'][79]
    df_collapsed[79]
    df_collapsed.shape
    type(df_collapsed)
    df.shape

    # # looking for occurrence of 'ttt' string
    ttt = df_collapsed.str.contains('ttt')
    len(ttt[ttt==True])
    # ttt[ttt==True]
    # df_collapsed[333]
    # df_collapsed[79]

    # # Create Wordcloud from data with stripped out html
    # wc.create_wordcloud_from_df(df_collapsed)


    # df_collapsed.isnull().sum()
    # ## There are 110 NaN values after cleaning, solved!

    # Create series as mask for nan values
    nulls = pd.isnull(df_collapsed)
    # nulls[nulls == True].index[0]
    # df_collapsed[79]
    # df['question'][79]




    '''
    Trying out different Topic Modeling alg's
    '''
    NUM_TOPICS = 5

    tfidf_vectorizer = TfidfVectorizer(min_df=5, max_df=0.80, stop_words='english', lowercase=True, token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')

    count_vectorizer = CountVectorizer(min_df=5, max_df=0.80, stop_words='english', lowercase=True, token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')

    data_tfidf_vectorized = tfidf_vectorizer.fit_transform(df_collapsed)
    # feature_names = tfidf_vectorizer.get_feature_names()
    data_count_vectorized = count_vectorizer.fit_transform(df_collapsed)

    type(data_count_vectorized)

    # print(feature_names[8])

    # Build a Latent Dirichlet Allocation Model
    lda_model = LatentDirichletAllocation(n_components=NUM_TOPICS, max_iter=10, learning_method='online')
    lda_Z = lda_model.fit_transform(data_count_vectorized)
    print(lda_Z.shape)  # (NO_DOCUMENTS, NO_TOPICS)
    print(lda_Z[0])

    # Build a Non-Negative Matrix Factorization Model
    nmf_model = NMF(n_components=NUM_TOPICS)
    nmf_Z = nmf_model.fit_transform(data_tfidf_vectorized)
    print(nmf_Z.shape)  # (NO_DOCUMENTS, NO_TOPICS)

    # Build a Latent Semantic Indexing Model
    lsi_model = TruncatedSVD(n_components=NUM_TOPICS)
    lsi_Z = lsi_model.fit_transform(data_tfidf_vectorized)
    print(lsi_Z.shape)  # (NO_DOCUMENTS, NO_TOPICS)

    # Let's see how the first document in the corpus looks like in different topic spaces
    print(lda_Z[0])
    print(nmf_Z[0])
    print(lsi_Z[0])


    def print_topics(model, vectorizer, top_n=10):
        for idx, topic in enumerate(model.components_):
            print("Topic %d:" % (idx))
            print([(vectorizer.get_feature_names()[i], topic[i])
                            for i in topic.argsort()[:-top_n - 1:-1]])

    print("Latent Dirichlet Allocation Model:")
    print_topics(lda_model, vectorizer)
    print("=" * 20)

    print("NMF (Non-negative matrix factorization) Model:")
    print_topics(nmf_model, vectorizer)
    print("=" * 20)

    print("LSI Model:")
    print_topics(lsi_model, vectorizer)
    print("=" * 20)

    df.shape
    df_collapsed.shape
