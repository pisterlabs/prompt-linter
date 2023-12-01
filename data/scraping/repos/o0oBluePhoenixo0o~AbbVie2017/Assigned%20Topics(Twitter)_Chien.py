### Before running this script, please put this file into the data repository so as to run it.

import pandas as pd
import numpy as np
import os
import re
import csv
import sys
import time
import string
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import gensim
from gensim import corpora, models, similarities
from gensim.models import CoherenceModel, LdaModel, LsiModel, HdpModel
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pyLDAvis.gensim as gensimvis
import pyLDAvis
from IPython.display import Image
from IPython.display import display

### Load the corpus, doc_term_matrix to build topic model
df_postn = pd.read_csv('10000_twitter_preprocessing.csv', encoding = 'UTF-8', sep = ',', index_col = 0)
df_postn = df_postn.sort_values(['created_time'])
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
    normalized = " ".join(lemma.lemmatize(word, pos='n') for word in punc_free.split())
    normalized = " ".join(lemma.lemmatize(word, pos='v') for word in normalized.split())
    word = " ".join(word for word in normalized.split() if len(word) > 3)
    postag = nltk.pos_tag(word.split())
    poslist = ['NN', 'NNP', 'NNS', 'RB', 'RBR', 'RBS', 'JJ', 'JJR', 'JJS']
    wordlist = ['co', 'https', 'http', 'rt', 'www', 've', 'dont', "i'm", "it's",'kf4pdwe64k','co', 'https', 'http', 'rt', 'www', 've', 'dont', "i'm", "it's",'kf4pdwe64k','nhttps','cant','didnt']
    adjandn = [word for word, pos in postag if pos in poslist and word not in wordlist and len(word) > 3]
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

corpus = list(df_postn['re_message'])
directory = "doc_clean.txt"
if os.path.exists(directory):
    with open("doc_clean.txt", "rb") as fp:  # Unpickling
        doc_clean = pickle.load(fp)
else:
    doc_clean = [tokenize(doc).split() for doc in corpus]
    with open("doc_clean.txt", "wb") as fp:  #Pickling
        pickle.dump(doc_clean, fp)
directory = "corpus.dict"
if os.path.exists(directory):
    dictionary = corpora.Dictionary.load('corpus.dict')
else:
    dictionary = corpora.Dictionary(doc_clean)
    dictionary.save('corpus.dict')
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
tfidf = models.TfidfModel(doc_term_matrix)
finalcorpus=tfidf[doc_term_matrix]

### Load original Twitter dataset to assign topics based on pre-trained LDA model with K=30
dk = pd.read_csv('TW_Tweet.csv', encoding='UTF-8', low_memory=False)
df = pd.DataFrame(dk, columns=['id', 'keyword', 'created', 'language', 'message'])
df.columns = ['id', 'key', 'created_time', 'language', 'message']
rm_duplicates = df.drop_duplicates(subset = ['key', 'message'])
rm_na = rm_duplicates.dropna()
dtime = rm_na.sort_values(['created_time'])
dtime.index = range(len(dtime))
dlang = dtime[dtime['language'] == 'en']
data = dlang[dlang['key'] != 'johnson & johnson']
data = data[data['key'] != 'johnson&johnson']
data.index = range(len(data))
# ldamodel = LdaModel(finalcorpus, num_topics = 30, id2word = dictionary, update_every = 10, chunksize=2000, passes=10, alpha=0.05)
# ldamodel.save('lda30.model')
ldamodel = LdaModel.load('lda30.model')
vis_data = gensimvis.prepare(ldamodel, finalcorpus, dictionary)
# pyLDAvis.save_html(vis_data, 'lda30.html')
vistopicid = vis_data[6]
idlist = []
for j in range(1, len(vistopicid) + 1):
    idlist.append([i for i, x in enumerate(vistopicid) if x == j][0])
topicwords = {}
no = 0
for prob in ldamodel.show_topics(30,7):
    tokens = ' '.join(re.findall(r"[\w']+", str(prob[1]))).lower().split()
    x = [''.join(c for c in s if c not in string.punctuation) for s in tokens]
    result = ' '.join([i for i in x if not i.isdigit()])
    topicwords[idlist[no]] = result.split()
    no += 1
for i in range(30):
    print("Topic", i+1, ": ", topicwords[i])

### Assign topics
'''
Self-defined function to assgin topics to each tweet in the dataset
The function return three results
* getTopicForQuery_lda(question)[0]: probability
* getTopicForQuery_lda(question)[1]: topic_id
* getTopicForQuery_lda(question)[2]: topic contents
'''
def getTopicForQuery_lda(question):
    temp = tokenize(question).split()
    ques_vec = []
    ques_vec = dictionary.doc2bow(temp)
    topic_vec = []
    topic_vec = ldamodel[ques_vec]
    word_count_array = np.empty((len(topic_vec), 2), dtype = np.object)
    for i in range(len(topic_vec)):
        word_count_array[i, 0] = topic_vec[i][0]
        word_count_array[i, 1] = topic_vec[i][1]
    idx = np.argsort(word_count_array[:, 1])
    idx = idx[::-1]
    word_count_array = word_count_array[idx]
    final = []
    final = ldamodel.print_topic(word_count_array[0, 0], 100)
    question_topic = final.split('*') # as format is like "probability * topic"
    tokens = ' '.join(re.findall(r"[\w']+", str(final))).lower().split()
    x = [''.join(c for c in s if c not in string.punctuation) for s in tokens]
    result = ' '.join([i for i in x if not i.isdigit()])
    topic_prob = list(reversed(sorted(ldamodel.get_document_topics(ques_vec), key = lambda tup: tup[1])))
    topic_prob = [list(t) for t in topic_prob]
    for i in range(len(topic_prob)):
        topic_prob[i][0] = idlist[topic_prob[i][0]] + 1
    return topic_prob[0][1], idlist[word_count_array[0, 0]] + 1, result.split()[0:10]
    
### Open a csv file to save the results in    
with open('twitter_topic_final.csv', 'w', encoding = 'UTF-8', newline = '') as csvfile:
    column = [['id', 'key', 'created_time', 'message', 'topic_id', 'probability', 'topic']]
    writer = csv.writer(csvfile)
    writer.writerows(column)
for i in range(len(data)):
    features = []
    features.append(data['id'][i])
    features.append(data['key'][i])
    features.append(data['created_time'][i])
    features.append(data['message'][i])
    question = data["message"][i]
    features.append(getTopicForQuery_lda(question)[1])
    features.append(getTopicForQuery_lda(question)[0])
    features.append(', '.join(getTopicForQuery_lda(question)[2]))
    with open('twitter_topic_final.csv', 'a', encoding = 'UTF-8', newline = '') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows([features])

test = pd.read_csv('twitter_topic_final.csv', encoding = 'UTF-8', sep = ',')
print("Overall Probability:", sum(test['probability'])/len(test))

### Load testing dataset to assign topics based on pre-trained LDA model with K=30
### This is the file created for manually label dataset with sample size=1000
dk = pd.read_csv('testingset.csv', encoding = 'ISO-8859-2', low_memory = False, index_col = 0)
data = pd.DataFrame(dk, columns = ['id', 'created_time', 'message'])
data.columns = ['id', 'created_time', 'message']
data.index = range(len(data))
# ldamodel = LdaModel(finalcorpus, num_topics = 30, id2word = dictionary, update_every = 10, chunksize=2000, passes=10, alpha=0.05)
# ldamodel.save('lda30.model')
ldamodel = LdaModel.load('lda30.model')
vis_data = gensimvis.prepare(ldamodel, finalcorpus, dictionary)
vistopicid = vis_data[6]
idlist = []
for j in range(1, len(vistopicid) + 1):
    idlist.append([i for i, x in enumerate(vistopicid) if x == j][0])
topicwords = {}
no = 0
for prob in ldamodel.show_topics(30,10):
    tokens = ' '.join(re.findall(r"[\w']+", str(prob[1]))).lower().split()
    x = [''.join(c for c in s if c not in string.punctuation) for s in tokens]
    result = ' '.join([i for i in x if not i.isdigit()])
    topicwords[idlist[no]]=result.split()
    no += 1

### Open a csv file to save the results in 
with open('lda_testingset.csv', 'w', encoding = 'UTF-8', newline = '') as csvfile:
    column = [['id', 'created_time', 'message', 'topic_id', 'probability', 'topic']]
    writer = csv.writer(csvfile)
    writer.writerows(column)
for i in range(len(data)):
    features = []
    features.append(data['id'][i])
    features.append(data['created_time'][i])
    features.append(data['message'][i])
    question = data["message"][i]
    features.append(getTopicForQuery_lda(question)[1])
    features.append(getTopicForQuery_lda(question)[0])
    features.append(', '.join(getTopicForQuery_lda(question)[2]))
    with open('lda_testingset.csv', 'a', encoding = 'UTF-8', newline = '') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows([features])

sample = pd.read_csv('lda_testingset.csv', encoding = 'UTF-8', sep = ',')
print("Overall Probability:", sum(sample['probability'])/len(sample))

