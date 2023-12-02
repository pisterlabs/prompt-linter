import re
from pprint import pprint
import pickle
import pandas as pd

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

from nltk.corpus import stopwords
import spacy  # spacy for lemmatization

import sys
# Enable logging for gensim - optional
import logging
import csv
import warnings

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

warnings.filterwarnings("ignore", category=DeprecationWarning)


def nlp(cluster_id=-1):
    # NLTK Stop words
    # 5. Prepare Stopwords
    stop_words = stopwords.words('english')
    stop_words.extend(['from', 'subject', 're', 'edu', 'use'])  # used for email, since the toy dataset is email set

    def sent_to_words(sentences):
        for sentence in sentences:
            yield gensim.utils.simple_preprocess(str(sentence), deacc=True)  # deacc=True removes punctuations

    # Define functions for stopwords, bigrams, trigrams and lemmatization
    def remove_stopwords(textss):
        return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in textss]

    def make_bigrams(textss):
        return [bigram_mod[doc] for doc in textss]

    def make_trigrams(textss):
        return [trigram_mod[bigram_mod[doc]] for doc in textss]

    def lemmatization(textss, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        """https://spacy.io/api/annotation"""
        texts_out = []
        for sent in textss:
            doc = nlp(" ".join(sent))
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        return texts_out

    # Import Newsgroups Data
    # df = pd.read_json('https://raw.githubusercontent.com/selva86/datasets/master/newsgroups.json')
    # df = pd.read_csv("yt_videos_data_news_politics_chunks/yt_videos_data_news_politics_chunk_3.csv")
    # create dataset
    ll = []
    for i in range(365):
        ll.append(pd.read_csv("yt_videos_data_news_politics/yt_videos_data_news_politics_{}.csv".format(i)))

    df = pd.concat(ll, ignore_index=True)
    df.dropna(inplace=True)
    # find videos belonging to the same cluster, otherwise process all videos
    if cluster_id != -1:
        # find correct videos basing on cluster id
        df_ids = pd.read_csv("labeled_communities_weighted.csv")
        # # select id of cluster
        df_ids = df_ids[df_ids["cluster"] == cluster_id]
        ids = df_ids["video_id"].values.tolist()
        df = df[df["display_id"].isin(ids)]

    # data = df.description.values.tolist()
    data = df.title.values.tolist() # use only title
    # 7. Remove emails and newline characters
    # remove web links
    # data = [re.sub(r'^https?:\/\/.*[\r\n]*', '', sent) for sent in data] # todo test if works
    data = [re.sub(r'http\S+', '', sent) for sent in data]  # todo test if works
    # Remove Emails
    data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]
    # Remove new line characters
    data = [re.sub('\s+', ' ', sent) for sent in data]
    # Remove distracting single quotes
    data = [re.sub("\'", "", sent) for sent in data]
    # Remove annoying line | in the title
    data = [re.sub("\|", "", sent) for sent in data]
    out_words = open('lda-data/data', 'wb')  # save in order to avoid recomputing
    pickle.dump(data, out_words)
    out_words.close()
    print("finish 7")

    # 8. Tokenize words and Clean-up text
    data_words = list(sent_to_words(data))
    print("finish 8")

    # 9. Build the bigram and trigram models
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)  # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)
    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    print("finish 9")

    # 10. Remove Stopwords, Make Bigrams and Lemmatize
    # Remove Stop Words
    data_words_nostops = remove_stopwords(data_words)
    # Form Bigrams
    data_words_bigrams = make_bigrams(data_words_nostops)
    # Initialize spacy 'en_core_web_sm' model, keeping only tagger component (for efficiency)
    # python3 -m spacy download en_core_web_sm
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    print("nlp")
    # Do lemmatization keeping only noun, adj, vb, adv
    data_lemmatized = lemmatization(data_words_bigrams,
                                    allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])  # time comsuming here
    print("lemmatizing")
    print(data_lemmatized[:1])
    print("finish 10")


    # 11. Create the Dictionary and Corpus needed for Topic Modeling
    # Create Dictionary
    id2word = corpora.Dictionary(data_lemmatized)
    out_words = open('lda-data/words-cluster', 'wb')  # save in order to avoid recomputing
    pickle.dump(id2word, out_words)
    out_words.close()
    # Create Corpus
    texts = data_lemmatized
    out_lemmatized = open('lda-data/lemmatized-cluster', 'wb')
    pickle.dump(data_lemmatized, out_lemmatized)
    out_lemmatized.close()
    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]
    out_corpus = open('lda-data/corpus-cluster', 'wb')
    pickle.dump(corpus, out_corpus)
    out_corpus.close()
    print("11 ended")

    return data, id2word, data_lemmatized, corpus
