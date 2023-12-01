import pandas as pd
import numpy as np
import re
import nltk
import csv
import sys
#from nltk import wordpunct_tokenize
from nltk.corpus import stopwords
import time
import os
import pyLDAvis.gensim as gensimvis
import pyLDAvis

#os.remove('./ML/Python/static/final_twitter_preprocessing_0720.csv')
#os.remove('./ML/Python/static/twitter_preprocessing_0720.csv')
start_time = time.time()
#disease = pd.read_csv('Final_TW_0807_prep.csv', encoding='ISO-8859-2', low_memory=False)
#df = pd.DataFrame(disease, columns = ['Id','key','created_time','Language', 'message'])
disease = pd.read_csv(sys.argv[1], encoding='UTF-8', low_memory=False)
df = pd.DataFrame(
    disease, columns=['id', 'keyword', 'created', 'language', 'message'])
df.columns = ['id', 'key', 'created_time', 'language', 'message']
rm_duplicates = df.drop_duplicates(subset=['key', 'message'])
rm_na = rm_duplicates.dropna()
dtime = rm_na.sort_values(['created_time'])
dtime.index = range(len(dtime))
dlang = dtime[dtime['language'] == 'en']
dlang.index = range(len(dlang))

import gensim
from gensim import corpora, models, similarities
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.stem import WordNetLemmatizer
import string
import time
stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()

with open(
        './ML/Python/topic/lda/static/twitter_preprocessing_0720.csv',
        'w',
        encoding='UTF-16LE',
        newline='') as csvfile:
    column = [[
        'id', 'key', 'created_time', 'language', 'message', 're_message'
    ]]
    writer = csv.writer(csvfile)
    writer.writerows(column)
for i in range(len(dlang['message'])):
    features = []
    features.append(str(int(dlang['id'][i])))
    features.append(dlang['key'][i])
    features.append(dlang['created_time'][i])
    features.append(dlang['language'][i])
    features.append(dlang['message'][i])
    k = dlang['message'][i].split()
    tokens = ' '.join(
        re.findall(r"[\w']+", str(dlang['message'][i]))).lower().split()
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
    irlist = [
        ',', '.', ':', '#', ';', 'CD', 'WRB', 'RB', 'PRP', '...', ')', '(',
        '-', '``', '@'
    ]
    poslist = ['NN', 'NNP', 'NNS', 'RB', 'RBR', 'RBS', 'JJ', 'JJR', 'JJS']
    wordlist = [
        'co', 'https', 'http', 'rt', 'com', 'amp', 'fe0f', 'www', 've', 'dont',
        "i'm", "it's", 'isnt', 'âźă', 'âąă', 'âł_'
    ]
    adjandn = [
        word for word, pos in postag
        if pos in poslist and word not in wordlist and len(word) > 3
    ]
    stop = set(stopwords.words('english'))
    wordlist = [i for i in adjandn if i not in stop]
    #print(wordlist)
    features.append(' '.join(wordlist))
    with open(
            './ML/Python/topic/lda/static/twitter_preprocessing_0720.csv',
            'a',
            encoding='UTF-16LE',
            newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows([features])
df_postncomment = pd.read_csv(
    './ML/Python/topic/lda/static/twitter_preprocessing_0720.csv',
    encoding='UTF-16LE',
    sep=',')
df_rm = df_postncomment.drop_duplicates(subset=['id', 're_message'])
rm_english_na = df_rm.dropna()
rm_english_na.index = range(len(rm_english_na))
dfinal_tw = pd.DataFrame(
    rm_english_na,
    columns=['id', 'created_time', 'language', 'message', 're_message'])
dfinal_tw.to_csv(
    './ML/Python/topic/lda/static/final_twitter_preprocessing_0720.csv',
    encoding='UTF-16LE',
    columns=['id', 'created_time', 'language', 'message', 're_message'])

import gensim
from gensim import corpora, models, similarities
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.stem import WordNetLemmatizer
import string
import time
start_time = time.time()
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


import datetime
import dateutil.relativedelta


def dateselect(day):
    d = datetime.datetime.strptime(str(datetime.date.today()), "%Y-%m-%d")
    d2 = d - dateutil.relativedelta.relativedelta(days=day)
    df_time = df_postn['created_time']
    df_time = pd.to_datetime(df_time)
    mask = (df_time > d2) & (df_time <= d)
    period = df_postn.loc[mask]
    return period


import pickle
corpus = list(df_postn['re_message'])
doc_clean = [tokenize(doc).split() for doc in corpus]
with open("./ML/Python/topic/lda/static/doc_clean.txt", "wb") as fp:  #Pickling
    pickle.dump(doc_clean, fp)
dictionary = corpora.Dictionary(doc_clean)
dictionary.save('./ML/Python/topic/lda/static/corpus.dict')
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
tfidf = models.TfidfModel(doc_term_matrix)
finalcorpus = tfidf[doc_term_matrix]
from gensim.models import CoherenceModel, LdaModel, LsiModel, HdpModel
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

###From here is testing
ldamodel = LdaModel.load('./ML/Python/topic/lda/static/lda.model')
vis_data = gensimvis.prepare(ldamodel, finalcorpus, dictionary)
#pyLDAvis.save_html(vis_data, 'static_lda_result.html')
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
for i in range(30):
    print("Topic", i + 1, ": ", topicwords[i])
