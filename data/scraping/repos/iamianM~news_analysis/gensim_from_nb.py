import os
import re
import operator
import matplotlib.pyplot as plt
import warnings
import gensim
import numpy as np
warnings.filterwarnings('ignore')  # Let's not pay heed to them right now

import nltk
nltk.download('stopwords') # Let's make sure the 'stopword' package is downloaded & updated
nltk.download('wordnet') # Let's also download wordnet, which will be used for lemmatization

from gensim.models import CoherenceModel, LdaModel, LsiModel, HdpModel
from gensim.models.wrappers import LdaMallet
from gensim.corpora import Dictionary
from pprint import pprint

import numpy as np
import pandas as pd

import pyLDAvis.gensim

df = pd.read_csv('../data/rss_feeds_new.csv')
df = df[pd.notnull(df['processed_text'])]
processed_text = df['processed_text'].values.tolist()

def build_texts(text):
    """
    Function to build tokenized texts from file

    Parameters:
    ----------
    fname: File to be read

    Returns:
    -------
    yields preprocessed line
    """
    for line in text:
        yield gensim.utils.simple_preprocess(line, deacc=True, min_len=3)

train_texts = list(build_texts(processed_text))

bigram = gensim.models.Phrases(train_texts)  # for bigram collocation detection

from gensim.utils import lemmatize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def process_texts(article_text):
    """
    Function to process texts. Following are the steps we take:

    1. Stopword Removal.
    2. Collocation detection.
    3. Lemmatization (not stem since stemming can reduce the interpretability).

    Parameters:
    ----------
    texts: Tokenized texts.

    Returns:
    -------
    texts: Pre-processed tokenized texts.
    """
#     article_text = [[word for word in line if word not in stops] for line in article_text]

    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()

    quotes1 = ' '.join(re.findall('“.*?”', article_text))
    quotes2 = ' '.join(re.findall('".*?"', article_text))
    quotes = quotes1 + quotes2

    tweets = ' '.join(re.findall('\n\n.*?@', article_text))+' '+' '.join(re.findall('\n\n@.*?@', article_text))

    article_text = re.sub('\n\n.*?@', '', article_text)
    article_text = re.sub('\n\n@.*?@', '', article_text)
    # Remove tweet
    article_text = ' '.join([word for word in article_text.split(' ') if not word.startswith('(@') and not word.startswith('http')])

    article_text = re.sub('“.*?”', '', article_text)
    article_text = re.sub('".*?"', '', article_text)

    from nltk.tokenize import RegexpTokenizer
    tokenizer = RegexpTokenizer(r'\w+')
    # create English stop words list
    sw = set(stopwords.words('english'))
    wordnet = WordNetLemmatizer()

    article_text = article_text.lower()
    quotes = quotes.lower()
    tweets = tweets.lower()

    article_text_tokens = tokenizer.tokenize(article_text)
    quotes_tokens = tokenizer.tokenize(quotes)
    tweets_tokens = tokenizer.tokenize(tweets)

    # remove stop words, unwanted words, tweet handles, links from tokens
    article_text_stopped_tokens = [i for i in article_text_tokens if i not in sw and i not in words_to_remove]
    quotes_stopped_tokens = [i for i in quotes_tokens if not i in sw and i not in words_to_remove]
    tweets_stopped_tokens = [i for i in tweets_tokens if not i in sw and i not in words_to_remove]

    article_text_stopped_tokens = bigram[article_text_stopped_tokens]
    quotes_stopped_tokens = bigram[quotes_stopped_tokens]
    tweets_stopped_tokens = bigram[tweets_stopped_tokens]

    # stem token
    article_text = [wordnet.lemmatize(i) for i in article_text_stopped_tokens]
    quotes = [wordnet.lemmatize(i) for i in quotes_stopped_tokens]
    tweets = [wordnet.lemmatize(i) for i in tweets_stopped_tokens]

    return article_text, quotes, tweets

def gensim_lda(df, n_topics):
    df = df[pd.notnull(df['article_text'])]
    text = df['article_text'].values.tolist()

    train_texts = list(build_texts(processed_text))
    bigram = gensim.models.Phrases(train_texts)  # for bigram collocation detection

    train_texts = []
    for article in text:
        temp = process_texts(article)
        train_texts.append(temp[0] + temp[1] + temp[2])

    dictionary = Dictionary(train_texts)
    corpus = [dictionary.doc2bow(text) for text in train_texts]

    chunksize = 10000
    passes = 50         # Was 20
    iterations = 400     # Was 400
    eval_every = None  # Don't evaluate model perplexity, takes too much time.

    ldamodel = LdaModel(corpus=corpus, num_topics=n_topics, id2word=dictionary, chunksize=chunksize, \
                   alpha='auto', eta='auto', \
                   iterations=iterations, \
                   passes=passes, eval_every=eval_every)
    ldatopics = ldamodel.show_topics(formatted=False)

    vis_data = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)
    pyLDAvis.save_html(vis_data, '../plots/pyLDAvis_'+str(n_topics)+'topics_gensim.html')

    return ldamodel


train_texts = []
for article in text:
    temp = process_texts(article)
    train_texts.append(temp[0] + temp[1] + temp[2])

dictionary = Dictionary(train_texts)
corpus = [dictionary.doc2bow(text) for text in train_texts]

lsimodel = LsiModel(corpus=corpus, num_topics=10, id2word=dictionary)
lsitopics = lsimodel.show_topics(formatted=False)

hdpmodel = HdpModel(corpus=corpus, id2word=dictionary)
hdptopics = hdpmodel.show_topics(formatted=False)

ldamodel = LdaModel(corpus=corpus, num_topics=10, id2word=dictionary)
ldatopics = ldamodel.show_topics(formatted=False)


def evaluate_graph(dictionary, corpus, texts, limit):
    """
    Function to display num_topics - LDA graph using c_v coherence

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    limit : topic limit

    Returns:
    -------
    lm_list : List of LDA topic models
    c_v : Coherence values corresponding to the LDA model with respective number of topics
    """
    chunksize = 10000
    passes = 1         # Was 20
    iterations = 50     # Was 400
    eval_every = None  # Don't evaluate model perplexity, takes too much time.


    c_v = []
    lm_list = []
    for num_topics in range(55, limit+1, 5):
        print(num_topics)
        lm = LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary, chunksize=chunksize, \
                       alpha='auto', eta='auto', \
                       iterations=iterations, \
                       passes=passes, eval_every=eval_every)
        lm_list.append(lm)
        cm = CoherenceModel(model=lm, texts=texts, dictionary=dictionary, coherence='c_v')
        c_v.append(cm.get_coherence())

    return lm_list, c_v

lmlist, c_v = evaluate_graph(dictionary=dictionary, corpus=corpus, texts=train_texts, limit=80)

def plot_coherence(c_v, labels):
    x = range(1, len(c_v))
    plt.plot(x, c_v_all)
    plt.xticks(x, labels)
    plt.xlabel("num_topics")
    plt.ylabel("Coherence score")
    plt.legend(("c_v"), loc='best')
    plt.show()

pyLDAvis.gensim.prepare(lmlist_all[11], corpus, dictionary)

def ret_top_model():
    """
    Since LDAmodel is a probabilistic model, it comes up different topics each time we run it. To control the
    quality of the topic model we produce, we can see what the interpretability of the best topic is and keep
    evaluating the topic model until this threshold is crossed.

    Returns:
    -------
    lm: Final evaluated topic model
    top_topics: ranked topics in decreasing order. List of tuples
    """
    top_topics = [(0, 0)]
    while top_topics[0][1] < 0.97:
        lm = LdaModel(corpus=corpus, id2word=dictionary)
        coherence_values = {}
        for n, topic in lm.show_topics(num_topics=-1, formatted=False):
            topic = [word for word, _ in topic]
            cm = CoherenceModel(topics=[topic], texts=train_texts, dictionary=dictionary, window_size=10)
            coherence_values[n] = cm.get_coherence()
        top_topics = sorted(coherence_values.items(), key=operator.itemgetter(1), reverse=True)
    return lm, top_topics

lm, top_topics = ret_top_model()

lda_lsi_topics = [[word for word, prob in lm.show_topic(topicid)] for topicid, c_v in top_topics]

lsitopics = [[word for word, prob in topic] for topicid, topic in lsitopics]
hdptopics = [[word for word, prob in topic] for topicid, topic in hdptopics]
ldatopics = [[word for word, prob in topic] for topicid, topic in ldatopics]
lmtopics = [[word for word, prob in topic] for topicid, topic in lmtopics]

lsi_coherence = CoherenceModel(topics=lsitopics[:10], texts=train_texts, dictionary=dictionary, window_size=10).get_coherence()
hdp_coherence = CoherenceModel(topics=hdptopics[:10], texts=train_texts, dictionary=dictionary, window_size=10).get_coherence()
lda_coherence = CoherenceModel(topics=ldatopics, texts=train_texts, dictionary=dictionary, window_size=10).get_coherence()
lm_coherence = CoherenceModel(topics=lmtopics, texts=train_texts, dictionary=dictionary, window_size=10).get_coherence()
lda_lsi_coherence = CoherenceModel(topics=lda_lsi_topics[:10], texts=train_texts, dictionary=dictionary, window_size=10).get_coherence()

def evaluate_bar_graph(coherences, indices):
    """
    Function to plot bar graph.

    coherences: list of coherence values
    indices: Indices to be used to mark bars. Length of this and coherences should be equal.
    """
    assert len(coherences) == len(indices)
    n = len(coherences)
    x = np.arange(n)
    plt.bar(x, coherences, width=0.2, tick_label=indices, align='center')
    plt.xlabel('Models')
    plt.ylabel('Coherence Value')

evaluate_bar_graph([lsi_coherence, hdp_coherence, lda_coherence, lm_coherence, lda_lsi_coherence],
                   ['LSI', 'HDP', 'LDA', 'LDA_Mod', 'LDA_LSI'])
