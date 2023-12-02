# from click import group
# from gensim.models import CoherenceModel
from gensim.utils import simple_preprocess
# from gensim.test.utils import datapath
from nltk.stem import WordNetLemmatizer
# from pprint import pprint
import nltk
from nltk.corpus import stopwords
from gensim.matutils import cossim
from datetime import date, timedelta, datetime

import gensim.corpora as corpora
# import os
import pandas as pd
# import re
import gensim
import requests
# import json

def run(today = ''):
    from reco_group import users_group
    if today == '':
        today = datetime.date(datetime.today() - timedelta(days = 1) + timedelta(hours = 2))
        today = datetime.strftime(today, '%Y-%m-%d')
    d = "http://api:8000/articles/{}".format(today)
    req = requests.get(d)

    j_data = req.json()
    if j_data != []:
        title = []
        text = []
        url = []
        
        for i in j_data:
            title.append(i['title'])
            text.append(i['text'])
            url.append(i['url'])

    df = pd.DataFrame({'text': text, 'title':title, 'url': url})


    lem = WordNetLemmatizer()

    #simulates articles input
    #df = pd.read_csv('/home/alnaggar/PBL/data-1653249353296.csv')
    lda_model = gensim.models.LdaMulticore.load("./lda_model.model")

    def sent_to_words(sentences):
        for sentence in sentences:
            # deacc=True removes punctuations
            yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))



    data = df.text.values.tolist()
    data_words = list(sent_to_words(data))

    # remove stop words

    # Create Dictionary
    id2word = corpora.Dictionary(data_words)
    # Create Corpus
    texts = data_words
    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]
    # View
    print(corpus[:1][0][:30])

    #a function that creates group features

    def groups_processing(df):
        df = df.sort_values(by = 'group')
        feature = df['feature'].tolist()
        g_id = df['group'].tolist()
        
        temp = ''
        g_feature = []
        
        for y in list(df['group'].unique()):
            for z in range(0, len(feature) - 1):
                if y == g_id[z]:
                    temp = temp + feature[z] + ' '
            g_feature.append(temp)
            temp = ''
        return g_feature

    g = groups_processing(users_group)

    data_g = list(sent_to_words(g))

    # remove stop words

    # Create Dictionary
    id2word_g = corpora.Dictionary(data_g)
    # Create Corpus
    texts_g = data_g
    # Term Document Frequency
    corpus_g = [id2word_g.doc2bow(text) for text in texts_g]

    group_vecs = []
    for x in corpus_g:
        group_vecs.append(lda_model.get_document_topics(x))

    print(len(corpus_g))

    def generate_recommendations(df, corpus, groups):
        scores = []
        art = 0
        for x in corpus:
            art_vec = lda_model.get_document_topics(x)
            for y in group_vecs:
                score = cossim(y, art_vec)
                scores.append(score)
            
            max_score = max(scores)
            g_ind = scores.index(max_score)
            #send article url with group id
            d_send = {"groupid":"", "url":""}
            d_send['groupid'] = g_ind
            d_send['url'] = df['url'].loc[art]
            #print(d_send)
            send = requests.post('http://api:8000/recommend/', json = d_send)
            scores = []
            art += 1
        return

    generate_recommendations(df, corpus, group_vecs)