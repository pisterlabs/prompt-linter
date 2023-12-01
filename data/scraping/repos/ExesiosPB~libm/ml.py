from app import app
from app import routes
from app import pygrams

from flask import render_template
import subprocess
from subprocess import check_output

import requests
import re
import GetOldTweets3 as got
import numpy
import pandas as pd
import json

from eventregistry import *
from textblob import TextBlob
from flask import jsonify

import pickle
import bz2
import time
import os

from gensim import corpora
from gensim.models import LsiModel
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt

@app.route('/ml/news/<string:search_param>')
def newsSearchML(search_param):
    s = routes.newsSearch(search_param)

    with bz2.BZ2File('df2.pkl.bz2', 'wb') as pickle_file:
        pickle.dump(
            s,
            pickle_file,
            protocol=4)

    cmd = ['python', 'pygrams.py', '-ds=df2.pkl.bz2', '-th=body']
    p = subprocess.Popen(cmd,
                     stdout=subprocess.PIPE,
                     stderr=subprocess.PIPE,
                     stdin=subprocess.PIPE)

    out,err = p.communicate()

    if err is not None:
        print(err)

    timestr = time.strftime("%Y%m%d-%H%M%S")

    with open('popular-terms-news' + timestr + 'txt', "wb") as thisFile:
        thisFile.write(out)

    return out


@app.route('/ml/wordcloud/news/<string:search_param>')
def newsSearchMLWordCloud(search_param):
    s = routes.newsSearch(search_param)

    with bz2.BZ2File('df2.pkl.bz2', 'wb') as pickle_file:
        pickle.dump(
            s,
            pickle_file,
            protocol=4)

    cmd = ['python', 'pygrams.py', '-ds=df2.pkl.bz2', '-th=body', "-o='wordcloud'"]
    p = subprocess.Popen(cmd,
                     stdout=subprocess.PIPE,
                     stderr=subprocess.PIPE,
                     stdin=subprocess.PIPE)

    out,err = p.communicate()

    if err is not None:
        print(err)

    timestr = time.strftime("%Y%m%d-%H%M%S")

    with open('popular-terms-news-wordcloud' + timestr, "wb") as thisFile:
        thisFile.write(out)

    return out

@app.route('/ml/lsa/news/<string:search_param>')
def newsSearchMLLSA(search_param):
    s = routes.newsSearch(search_param)
    number_of_topics=5
    words=10
    path = ""
    file_name = "stoke.json"
    documents_list = []
    titles=[]
    with open( os.path.join(path, file_name) ,"r") as fin:
        for line in fin.readlines():
            text = line.strip()
            documents_list.append(text)
    print("Total Number of Documents:",len(documents_list))
    titles.append( text[0:min(len(text),80)] )

    tokenizer = RegexpTokenizer(r'\w+')
    en_stop = stopwords.words('english')
    custom_list_of_words_to_exclude = ['articles', 'authors', 'isAgency', 'name', 'type', 'author', 'uri', 'document', 'step', 'token', 'start', 'end', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'co', 'citi', 'stokesentinel', 'said', 'news']
    en_stop.extend(custom_list_of_words_to_exclude)

    p_stemmer = PorterStemmer()
    texts = []
    for i in documents_list:
        raw = i.lower()
        tokens = tokenizer.tokenize(raw)
        stopped_tokens = [i for i in tokens if not i in en_stop]
        stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
        texts.append(stemmed_tokens)

    dictionary = corpora.Dictionary(texts)
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in texts]

    aa = []

    lsamodel = LsiModel(doc_term_matrix, num_topics=number_of_topics, id2word = dictionary)
    thisModel = lsamodel.print_topics(num_topics=number_of_topics, num_words=words)

    return jsonify(values=thisModel)
