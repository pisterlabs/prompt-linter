from DAO.DAOFactory import DAOFactory
from UserPartitioning import UserPartitioningStrategyFactory
from Builder.ContentMarketBuilder import ContentMarketBuilder
from Tweet.ContentMarketTweet import ContentMarketTweet

import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt
from typing import Set, List, Tuple, Dict
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
from gensim.models.nmf import Nmf
import numpy as np
from operator import itemgetter
import pickle
from datetime import datetime

nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')


def preprocess(tweets: Set[ContentMarketTweet]) -> Tuple[List[List[str]], List[str]]:
    """Preprocess the tweets for NMF topic modelling.
    
    Returns two different ways of representing the same corpus. The first is a list of lists, where
    each inner list is a list of tokens in the tweet. The second is a lsit, where each string in the
    list is the preprocessed text of the tweet.
    """
    corpus = []
    gensim_texts = []
    for tweet in tweets:
        tokens = word_tokenize(tweet.content)

        # pos tag words
        tags = [tag[1] for tag in nltk.pos_tag(tokens)]

        # remove @user and !url
        i = 0
        new_tokens, new_tags = [], []
        while i < len(tokens):
            if i != len(tokens) - 1:
                if tokens[i] == "@" and tokens[i + 1] == "user":
                    i += 2
                elif tokens[i] == "!" and tokens[i + 1] == "url":
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    new_tags.append(tags[i])
                    i += 1
            else:
                new_tokens.append(tokens[i])
                new_tags.append(tags[i])
                i += 1  
        tokens, tags = new_tokens, new_tags

        # remove numbers
        # remove punctuation and emojis i.e. any word that isn't entirely consisted of alphanumeric 
        # characters
        # remove stopwords
        nums_regex = re.compile('[0-9]+')
        stop_words = stopwords.words('english')
        new_tokens, new_tags = [], []
        for i in range(len(tokens)):
            if not nums_regex.match(tokens[i]) and tokens[i].isalnum() \
                and tokens[i] not in stop_words:
                new_tokens.append(tokens[i])
                new_tags.append(tags[i])
        tokens, tags = new_tokens, new_tags

        # lemmatize words
        new_tokens = []
        for i in range(len(tokens)):
            lemmatized_token = lemmatize(tokens[i], tags[i])
            new_tokens.append(lemmatized_token)
        tokens = new_tokens

        # we can stop caring about the tags after this point
        # remove stopwords again
        stop_words = stopwords.words('english')
        tokens = [token for token in tokens if token not in stop_words]
        corpus.append(" ".join(tokens))
        gensim_texts.append(tokens)

    return gensim_texts, corpus


def lemmatize(text: str, tag) -> str:
    """Helper function for preprocess.
    
    Lemmatize the given <text> depending on the <tag>."""
    tag_dict = {"JJ": wordnet.ADJ,
                "NN": wordnet.NOUN,
                "VB": wordnet.VERB,
                "RB": wordnet.ADV}

    pos = tag_dict.get(tag[0:2], wordnet.NOUN)
    return WordNetLemmatizer().lemmatize(text, pos=pos)


def find_good_num_topics(texts: List[List[str]]) -> None:
    """Find a good number of topics to use for NMF, using gensim.
    Creates a plot of the coherence scores that shows us a potential good number of topics to use.

    Source: https://www.kaggle.com/code/rockystats/topic-modelling-using-nmf"""
    # Use Gensim's NMF to get the best num of topics via coherence score
    # Create a dictionary
    # In gensim a dictionary is a mapping between words and their integer id
    dictionary = Dictionary(texts)

    # Filter out extremes to limit the number of features
    # dictionary.filter_extremes(
    #     # no_below=3,  # commented this out
    #     # no_above=0.85,  # commented this out
    #     # keep_n=5000
    # )

    # Create the bag-of-words format (list of (token_id, token_count))
    corpus = [dictionary.doc2bow(text) for text in texts]

    # Create a list of the topic numbers we want to try
    topic_nums = list(np.arange(5, 75 + 1, 5))

    # Run the NMF model and calculate the coherence score
    # for each number of topics
    coherence_scores = []

    for num in topic_nums:
        print(str(num) + "... " + str(datetime.now()))
        nmf = Nmf(
            corpus=corpus,
            num_topics=num,
            id2word=dictionary,
            chunksize=2000,
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

        # Run the coherence model to get the score
        cm = CoherenceModel(
            model=nmf,
            texts=texts,
            dictionary=dictionary,
            coherence='c_v'
        )
        
        coherence_scores.append(round(cm.get_coherence(), 5))

    # Get the number of topics with the highest coherence score
    scores = list(zip(topic_nums, coherence_scores))
    best_num_topics = sorted(scores, key=itemgetter(1), reverse=True)[0][0]

    # Plot the results
    fig = plt.figure(figsize=(15, 7))

    plt.plot(
        topic_nums,
        coherence_scores,
        linewidth=3,
        color='#4287f5'
    )

    plt.xlabel("Topic Num", fontsize=14)
    plt.ylabel("Coherence Score", fontsize=14)
    plt.title('Coherence Score by Topic Number - Best Number of Topics: {}'.format(best_num_topics), 
              fontsize=18)
    plt.xticks(np.arange(5, max(topic_nums) + 1, 5), fontsize=12)
    plt.yticks(fontsize=12)

    plt.show()


def create_nmf_model(tweets: List[ContentMarketTweet], corpus: List[str], n_components: int):
    """Create a NMF model with <n_components> topics.
    
    Returns three things: the NMF model, the vectorizer used for the NMF model, and a dictionary
    mapping tweet_ids to the NMF topic."""
    print("Tf-idf...")
    vectorizer = TfidfVectorizer(
        stop_words="english", 
        # min_df=3, 
        # max_df=0.85, 
        # max_features=5000, 
        # ngram_range=(1, 2)
        )
    X = vectorizer.fit_transform(corpus)

    print("NMF...")
    nmf = NMF(
        n_components=n_components,
        init="nndsvd",  # this works best for sparse data, like what we have
        random_state=42,
        beta_loss="frobenius",
        max_iter=500
    ).fit(X)

    docweights = nmf.transform(vectorizer.transform(corpus))
    topics = list(docweights.argmax(axis=1))

    tweet_id_to_topic = {}  # a dictionary mapping each tweet_id to the corresponding topic
    for i in range(len(tweets)):
        tweet_id_to_topic[tweets[i].id] = topics[i]
    
    return nmf, vectorizer, tweet_id_to_topic


def plot_top_words(model, feature_names, n_top_words, title):
    """Creates a plot of the top 10 words for each topic in the <model>.
    
    Source: https://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html#sphx-glr-download-auto-examples-applications-plot-topics-extraction-with-nmf-lda-py"""
    fig, axes = plt.subplots(5, 6, figsize=(60, 20), sharex=True)
    axes = axes.flatten()
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[: -n_top_words - 1 : -1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f"Topic {topic_idx +1}", fontdict={"fontsize": 10})
        ax.invert_yaxis()
        ax.tick_params(axis="both", which="major", labelsize=7)
        for i in "top right left".split():
            ax.spines[i].set_visible(False)
        fig.suptitle(title, fontsize=20)

    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    plt.show()
