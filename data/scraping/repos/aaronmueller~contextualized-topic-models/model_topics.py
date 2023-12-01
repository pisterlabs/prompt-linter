import os
from littlebird import TweetReader, TweetTokenizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import gensim
from gensim import corpora, models, matutils
from gensim.models import CoherenceModel
from gensim.models.wrappers import LdaMallet

MALLET_PATH = "/home/amueller/Mallet/bin/mallet"

DOCS_DIR = "data"

def _confirm_org(tweet):
    # ensure tweet is not a RT or QT
    if 'retweeted_status' not in tweet.keys() and tweet['is_quote_status'] is False:
        return True
    else:
        return False

def print_topics(model, count_vectorizer, n_top_words):
    words = count_vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(model.components_):
        print("\Topic #{}:".format(topic_idx))
        print(" ".join([words[i] for i in topic.argsort()[:-n_top_words-1:-1]]))


# load data
# num_docs = 0
doc_num = 0
filtered_tweets = []
tokenizer = TweetTokenizer(stopwords=stopwords.words('english'))
for filename in os.listdir(DOCS_DIR):
    if not filename.endswith(".json.gz"):
        continue
    reader = TweetReader(os.path.join(DOCS_DIR, filename))
    for tweet in reader.read_tweets():
        if not _confirm_org(tweet):
            continue
        doc_num += 1
        if doc_num % 2 != 0:
            continue
        if tweet.get("truncated", False):
            text = tweet["extended_tweet"]["full_text"]
        else:
            text = tweet["text"]
        tokens = tokenizer.tokenize(text)
        filtered_tweets.append(' '.join(tokens))
#    num_docs += 1
#    if num_docs > 10000:
#        break
#    break

# topic modeling
num_topics = 50
num_words = 20
count_vectorizer = CountVectorizer(stop_words=stopwords.words('english'))
count_vectorizer.fit(filtered_tweets)
doc_word = count_vectorizer.transform(filtered_tweets).transpose()

corpus = matutils.Sparse2Corpus(doc_word)

word2id = dict((v, k) for v, k in count_vectorizer.vocabulary_.items())
id2word = dict((v, k) for k, v in count_vectorizer.vocabulary_.items())
dictionary = corpora.Dictionary()
dictionary.id2token = id2word
dictionary.token2id = word2id

ldamallet = LdaMallet(MALLET_PATH, corpus=corpus, num_topics=num_topics, id2word=id2word)
print(ldamallet.show_topics(formatted=False))

coherence = CoherenceModel(model=ldamallet, texts=filtered_tweets, coherence='c_npmi')
print("coherence:", coherence.get_coherence())
