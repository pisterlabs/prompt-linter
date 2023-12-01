# following https://towardsdatascience.com/topic-modeling-articles-with-nmf-8c6b2a227a45
# this runs two versions of an NMF model, one is BOW and uses gensim's model, the other is tf-idf and uses sklearn's model
# we use bow to get best number of topics and then find those topics with the tfidf nmf model
import os
import nltk.tokenize.casual
import re
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import string
import contractions
import pandas as pd
import gensim
from gensim import models, corpora
from gensim.models.nmf import Nmf
from gensim.models.coherencemodel import CoherenceModel
import numpy as np
import operator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

# function to process and tokenize the text
def process_text(text):
    text = nltk.tokenize.casual.casual_tokenize(text)
    text = [each.lower() for each in text]
    text = [re.sub('[0-9]+', '', each) for each in text]
    text = [contractions.fix(each) for each in text]
    text = [SnowballStemmer('english').stem(each) for each in text]
    text = [w for w in text if w not in string.punctuation]
    text = [w for w in text if w not in stop_words]
    text = [each for each in text if len(each) > 1]
    text = [each for each in text if ' ' not in each]
    return text
        

def find_best_num_topics(dictionary, corpus, topic_nums):
     # Run the nmf model and calculate the coherence score
    coherence_scores = []
    for num in topic_nums:
        nmf = Nmf(
            corpus=corpus,
            num_topics=num,
            id2word=dictionary,
            chunksize=10,
            passes=5,
            kappa=.1,
            minimum_probability=0.01,
            w_max_iter=300,
            w_stop_condition=0.0001,
            h_max_iter=100,
            h_stop_condition=0.001,
            eval_every=10,
            normalize=True,
            random_state=42
        )
        # print(nmf.print_topics(num, 5))
        # Run the coherence model to get the score
        cm = CoherenceModel(
            model=nmf,
            texts=texts,
            dictionary=dictionary,
            coherence='c_v'
        )
        coherence_scores.append(round(cm.get_coherence(), 5))
    return coherence_scores

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()


if __name__ == '__main__':
    stop_words = stopwords.words('english')
    files = os.listdir('./dnc_speeches')
    # iterate over the list getting each file 
    speeches = []
    for fle in files:
        # open the file and then call .read() to get the text 
        with open('./dnc_speeches/'+fle) as f:
            if(fle != 'alexandria_ocasio-cortex.txt'):
                text = f.read()
                speeches.append(text)
    texts = [process_text(each) for each in speeches]
    tfidf_vectorizer = TfidfVectorizer(
        min_df=3,
        max_df=0.85,
        max_features=1000,
        ngram_range=(1, 2),
        preprocessor=' '.join
    )   
    tfidf = tfidf_vectorizer.fit_transform(texts)
    dictionary = gensim.corpora.dictionary.Dictionary(texts)

    # Filter out extremes to limit the number of features
    dictionary.filter_extremes(
        no_below=3,
        no_above=0.85,
        keep_n=1000
    )
    # Create the bag-of-words format (list of (token_id, token_count))
    corpus = [dictionary.doc2bow(text) for text in texts]

    # Create a list of the topic numbers we want to try
    topic_nums = [3, 5, 10, 15, 20]
    coherence_scores = find_best_num_topics(dictionary, corpus, topic_nums)
    scores = list(zip(topic_nums, coherence_scores))
    print(scores)
    best_num_topics = sorted(scores, key=operator.itemgetter(1), reverse=True)[0][0]
    print(best_num_topics)
    bow_nmf = Nmf(
            corpus=corpus,
            num_topics=best_num_topics,
            id2word=dictionary,
        ) 
    print(bow_nmf.print_topics(best_num_topics, 5))
    tfidf_nmf = NMF(
        n_components=best_num_topics,
        init='random',
        random_state=0
    ).fit(tfidf)
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()
    print_top_words(tfidf_nmf, tfidf_feature_names, 5)

    # test with a new speech, AOC's
    # Transform the new data with the fitted models
    with open('./dnc_speeches/alexandria_ocasio-cortex.txt') as f2:
        new_text = f2.read()
    new_text = process_text(new_text)
    tfidf_new = tfidf_vectorizer.transform(new_text)
    X_new = tfidf_nmf.transform(tfidf_new)
    # Get the top predicted topic
    predicted_topics = [np.argsort(each)[::-1][0] for each in X_new]
    print(predicted_topics)



