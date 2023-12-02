import sys
import re
import os
import json
import string
import gensim
import pandas as pd
import numpy as np
import nltk
import datetime
import dateutil.relativedelta
from nltk.corpus import stopwords
import pickle
from gensim import corpora, models, similarities
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.stem import WordNetLemmatizer
import pyLDAvis.gensim as gensimvis
import pyLDAvis

batchTweets = pd.read_csv(sys.argv[1], encoding='UTF-8', low_memory=False)
df = pd.DataFrame(
    batchTweets, columns=['id', 'keyword', 'created', 'language', 'message'])
df.columns = ['id', 'key', 'created_time', 'language', 'message']
df_postn = pd.read_csv(
    './ML/Python/topic/lda/static/final_twitter_preprocessing_0720.csv',
    encoding='UTF-16LE',
    sep=',',
    index_col=0)
df_postn.index = range(len(df_postn))

stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()


def tokenize(doc):
    reurl = re.sub(r"http\S+", "", str(doc))
    tokens = ' '.join(re.findall(r"[\w']+", reurl)).lower().split()
    x = [''.join(c for c in s if c not in string.punctuation) for s in tokens]
    x = ' '.join(x)
    stop_free = " ".join([i for i in x.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(
        lemma.lemmatize(word, pos='n') for word in punc_free.split())
    normalized = " ".join(
        lemma.lemmatize(word, pos='v') for word in normalized.split())
    word = " ".join(word for word in normalized.split() if len(word) > 3)
    postag = nltk.pos_tag(word.split())
    #irlist=[',','.',':','#',';','CD','WRB','RB','PRP','...',')','(','-','``','@']
    poslist = ['NN', 'NNP', 'NNS', 'RB', 'RBR', 'RBS', 'JJ', 'JJR', 'JJS']
    wordlist = [
        'co', 'https', 'http', 'rt', 'www', 've', 'dont', "i'm", "it's"
    ]
    adjandn = [
        word for word, pos in postag
        if pos in poslist and word not in wordlist and len(word) > 3
    ]
    return ' '.join(adjandn)


def dateselect(day):
    d = datetime.datetime.strptime(str(datetime.date.today()), "%Y-%m-%d")
    d2 = d - dateutil.relativedelta.relativedelta(days=day)
    df_time = df_postn['created_time']
    df_time = pd.to_datetime(df_time)
    mask = (df_time > d2) & (df_time <= d)
    period = df_postn.loc[mask]
    return period


corpus = list(df_postn['re_message'])

directory = "./ML/Python/topic/lda/static/doc_clean.txt"
if os.path.exists(directory):
    with open("./ML/Python/topic/lda/static/doc_clean.txt",
              "rb") as fp:  # Unpickling
        doc_clean = pickle.load(fp)
else:
    doc_clean = [tokenize(doc).split() for doc in corpus]
    with open("./ML/Python/topic/lda/static/doc_clean.txt",
              "wb") as fp:  #Pickling
        pickle.dump(doc_clean, fp)
directory = "./ML/Python/topic/lda/static/corpus.dict"
if os.path.exists(directory):
    dictionary = corpora.Dictionary.load(
        './ML/Python/topic/lda/static/corpus.dict')
else:
    dictionary = corpora.Dictionary(doc_clean)
    dictionary.save('./ML/Python/topic/lda/static/corpus.dict')
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
tfidf = models.TfidfModel(doc_term_matrix)
finalcorpus = tfidf[doc_term_matrix]
from gensim.models import CoherenceModel, LdaModel, LsiModel, HdpModel
directory = "./ML/Python/topic/lda/static/lda.model"
if os.path.exists(directory):
    ldamodel = LdaModel.load('./ML/Python/topic/lda/static/lda.model')
else:
    ldamodel = LdaModel(
        finalcorpus,
        num_topics=30,
        id2word=dictionary,
        update_every=10,
        chunksize=10000,
        passes=10,
        eta=None,
        alpha=0.05)
    ldamodel.save('./ML/Python/topic/lda/static/lda.model')

vis_data = gensimvis.prepare(ldamodel, finalcorpus, dictionary)
pyLDAvis.save_html(vis_data,
                   './ML/Python/topic/lda/static/static_lda_result.html')
vistopicid = vis_data[6]
idlist = []
for j in range(1, len(vistopicid) + 1):
    idlist.append([i for i, x in enumerate(vistopicid) if x == j][0])
topicwords = {}
no = 0
for prob in ldamodel.show_topics(30, 10):
    tokens = ' '.join(re.findall(r"[\w']+", str(prob[1]))).lower().split()
    x = [''.join(c for c in s if c not in string.punctuation) for s in tokens]
    result = ' '.join([i for i in x if not i.isdigit()])
    topicwords[idlist[no]] = result.split()
    no += 1


def getTopicForQuery_lda(question):
    """Extracts the topic for a specified message.

    Retrieves the topic for the specified message out of
    the pretrained LDA model

    Args:
        question: A string message

    Returns:
       
    """

    temp = tokenize(question).split()
    ques_vec = []
    ques_vec = dictionary.doc2bow(temp)
    topic_vec = []
    topic_vec = ldamodel[ques_vec]
    word_count_array = np.empty((len(topic_vec), 2), dtype=np.object)
    for i in range(len(topic_vec)):
        word_count_array[i, 0] = topic_vec[i][0]
        word_count_array[i, 1] = topic_vec[i][1]
    idx = np.argsort(word_count_array[:, 1])
    idx = idx[::-1]
    word_count_array = word_count_array[idx]
    final = []
    final = ldamodel.print_topic(word_count_array[0, 0], 100)
    question_topic = final.split(
        '*')  ## as format is like "probability * topic"
    tokens = ' '.join(re.findall(r"[\w']+", str(final))).lower().split()
    x = [''.join(c for c in s if c not in string.punctuation) for s in tokens]
    result = ' '.join([i for i in x if not i.isdigit()])
    topic_prob = list(
        reversed(
            sorted(
                ldamodel.get_document_topics(ques_vec),
                key=lambda tup: tup[1])))
    topic_prob = [list(t) for t in topic_prob]
    for i in range(len(topic_prob)):
        topic_prob[i][0] = idlist[topic_prob[i][0]] + 1
    return topic_prob[0][
        1], idlist[word_count_array[0, 0]] + 1, result.split()[0:10]


df_postn.index = range(len(df_postn))
k = []
for i in range(len(df)):
    tp_dict = {}
    question = df["message"][i]
    if str(df['id'][i]) == 'nan':
        tp_dict['key'] = 'nan'
    else:
        tp_dict['key'] = str(int(df['id'][i]))  #convert to string
    topic = getTopicForQuery_lda(question)
    tp_dict['id'] = topic[1]
    tp_dict['topic'] = ', '.join(topic[2])
    tp_dict['probability'] = topic[0]
    k.append(tp_dict)
print(json.dumps(k))
