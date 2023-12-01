# import datetime   ####### for saving cluster analytics
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# from matplotlib import cm
# import plotly as py
# import plotly.graph_objs as go

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
# my ORM
from my_declarative_base import Base, Images, Topics,ImagesTopics, Column, Integer, String, Date, Boolean, DECIMAL, BLOB, ForeignKey, JSON, ForeignKey

from sqlalchemy.exc import IntegrityError
from sqlalchemy import create_engine, text, MetaData, Table, Column, Numeric, Integer, VARCHAR, update, Float
from sqlalchemy.exc import OperationalError
from sqlalchemy.pool import NullPool
from pick import pick

import numpy as np
import pandas as pd
import os
import time
import pickle
from sys import platform

#mine
from mp_db_io import DataIO
###########
import gensim
from gensim.test.utils import get_tmpfile
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from gensim.models.coherencemodel import CoherenceModel
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import os

import nltk
from gensim import corpora, models
from pprint import pprint

#nltk.download('wordnet') ##only first time

# MM you need to use conda activate minimal_ds 

'''
tracking time based on items, for speed predictions
items, seconds
47000, 240
100000, 695
'''
title = 'Please choose your operation: '
options = ['Topic modelling', 'Topic indexing','calculating optimum_topics']
io = DataIO()
db = io.db
io.db["name"] = "ministock1023"

NUMBER_OF_PROCESSES = io.NUMBER_OF_PROCESSES
MODEL_PATH=os.path.join(io.ROOT,"model")
DICT_PATH=os.path.join(io.ROOT,"dictionary.dict")
BOW_CORPUS_PATH=os.path.join(io.ROOT,"BOW_lda_corpus.mm")
TFIDF_CORPUS_PATH=os.path.join(io.ROOT,"TFIDF_lda_corpus.mm")
# Satyam, you want to set this to False
USE_SEGMENT = False

MODEL="TF" ## OR TF  ## Bag of words or TF-IDF
NUM_TOPICS=88

stemmer = SnowballStemmer('english')


def set_query():
    # Basic Query, this works with gettytest3
    SELECT = "DISTINCT(image_id),description,keyword_list"
    FROM ="bagofkeywords"
    WHERE = "keyword_list IS NOT NULL "
    LIMIT = 2000000
    if MODE==1:
        WHERE = "keyword_list IS NOT NULL AND image_id NOT IN (SELECT image_id FROM imagestopics)"
        # WHERE = "image_id = 423638"
        LIMIT=100000
    return SELECT, FROM, WHERE, LIMIT

if db['unix_socket']:
    # for MM's MAMP config
    engine = create_engine("mysql+pymysql://{user}:{pw}@/{db}?unix_socket={socket}".format(
        user=db['user'], pw=db['pass'], db=db['name'], socket=db['unix_socket']
    ), poolclass=NullPool)
else:
    engine = create_engine("mysql+pymysql://{user}:{pw}@{host}/{db}"
                                .format(host=db['host'], db=db['name'], user=db['user'], pw=db['pass']), poolclass=NullPool)

#engine = create_engine("mysql+pymysql://{user}:{pw}@{host}/{db}".format(host=db['host'], db=db['name'], user=db['user'], pw=db['pass']), poolclass=NullPool)
# if MODE==0 or MODE==2:
    # WHERE = "keyword_list IS NOT NULL "
    # #LIMIT = 328894
    # LIMIT=100000
# elif MODE==1:
    # WHERE = "keyword_list IS NOT NULL AND image_id NOT IN (SELECT image_id FROM imagestopics)"
    # LIMIT=1000


# metadata = MetaData(engine)
Session = sessionmaker(bind=engine)
session = Session()
Base = declarative_base()

def selectSQL():
    SELECT, FROM, WHERE, LIMIT = set_query()
    selectsql = f"SELECT {SELECT} FROM {FROM} WHERE {WHERE} LIMIT {str(LIMIT)};"
    print("actual SELECT is: ",selectsql)
    result = engine.connect().execute(text(selectsql))
    resultsjson = ([dict(row) for row in result.mappings()])
    return(resultsjson)

def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result

# def save_model(lda_model,tfidf_corpus,bow_corpus):
    # CORPUS_PATH = get_tmpfile(CORPUS_PATH)
    # lda_model.save(MODEL_PATH)
    # print("model saved")
    # return

def gen_corpus(processed_txt,MODEL):
    dictionary = gensim.corpora.Dictionary(processed_txt)
    dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
    bow_corpus = [dictionary.doc2bow(doc) for doc in processed_txt] ## BOW corpus
    if MODEL=="TF":   
        tfidf = models.TfidfModel(bow_corpus)  ## converting BOW to TDIDF corpus
        tfidf_corpus = tfidf[bow_corpus]
    dictionary.save(DICT_PATH)
    corpora.MmCorpus.serialize(TFIDF_CORPUS_PATH, tfidf_corpus)
    corpora.MmCorpus.serialize(BOW_CORPUS_PATH, bow_corpus)

    return 
    
def LDA_model(num_topics):
    print("loading corpus and dictionary")
    loaded_dict = corpora.Dictionary.load(DICT_PATH)
    loaded_corp = corpora.MmCorpus(TFIDF_CORPUS_PATH)
    print("processing the model now")
    lda_model = gensim.models.LdaMulticore(loaded_corp, num_topics=num_topics, id2word=loaded_dict, passes=2, workers=NUMBER_OF_PROCESSES)
    lda_model.save(MODEL_PATH)
    print("processed all")
    return lda_model
    
def write_topics(lda_model):
    print("writing data to the topic table")
    for idx, topic_list in lda_model.print_topics(-1):
        print('Topic: {} \nWords: {}'.format(idx, topic_list))

       # Create a BagOfKeywords object
        topics_entry = Topics(
        topic_id = idx,
        topic = "".join(topic_list)
        )

    # Add the BagOfKeywords object to the session
        session.add(topics_entry)
        print("Updated topic_id {}".format(idx))
    session.commit()
    return

def write_imagetopics(resultsjson,lda_model_tfidf,dictionary):
    print("writing data to the imagetopic table")
    idx_list, topic_list = zip(*lda_model_tfidf.print_topics(-1))
    for i,row in enumerate(resultsjson):
        # print(row)
        keyword_list=" ".join(pickle.loads(row["keyword_list"]))

        # handles empty keyword_list
        if keyword_list:
            word_list = keyword_list
        else:
            word_list = row["description"]

        bow_vector = dictionary.doc2bow(preprocess(word_list))

        #index,score=sorted(lda_model_tfidf[bow_corpus[i]], key=lambda tup: -1*tup[1])[0]
        index,score=sorted(lda_model_tfidf[bow_vector], key=lambda tup: -1*tup[1])[0]
        imagestopics_entry=ImagesTopics(
        image_id=row["image_id"],
        topic_id=index,
        topic_score=score
        )
        session.add(imagestopics_entry)
        # print(f'image_id {row["image_id"]} -- topic_id {index} -- topic tokens {topic_list[index][:100]}')
        # print(f"keyword list {keyword_list}")

        if row["image_id"] % 1000 == 0:
            print("Updated image_id {}".format(row["image_id"]))


    # Add the imagestopics object to the session
    session.commit()
    return
def calc_optimum_topics(resultsjson):

    #######TOPIC MODELING ############
    txt = pd.DataFrame(index=range(len(resultsjson)),columns=["description","keywords","index","score"])
    for i,row in enumerate(resultsjson):
        #txt.at[i,"description"]=row["description"]
        txt.at[i,"keyword_list"]=" ".join(pickle.loads(row["keyword_list"]))
    #processed_txt=txt['description'].map(preprocess)
    processed_txt=txt['keyword_list'].map(preprocess)

    gen_corpus(processed_txt,MODEL)
    corpus = corpora.MmCorpus(BOW_CORPUS_PATH)
    dictionary = corpora.Dictionary.load(MODEL_PATH+'.id2word')

    
    num_topics_list=[80,90,100,110,120]
    coher_val_list=np.zeros(len(num_topics_list))
    for i,num_topics in enumerate(num_topics_list):
        lda_model = gensim.models.LdaMulticore(corpus, num_topics=num_topics, id2word=dictionary, passes=2, workers=NUMBER_OF_PROCESSES)
        cm = CoherenceModel(model=lda_model, corpus=corpus, coherence='u_mass')
        coher_val_list[i]=cm.get_coherence()
    print(num_topics_list,coher_val_list)  # get coherence value

def topic_model(resultsjson):
    #######TOPIC MODELING ############
    txt = pd.DataFrame(index=range(len(resultsjson)),columns=["description","keywords","index","score"])
    for i,row in enumerate(resultsjson):
        #txt.at[i,"description"]=row["description"]
        txt.at[i,"keyword_list"]=" ".join(pickle.loads(row["keyword_list"]))
    #processed_txt=txt['description'].map(preprocess)
    processed_txt=txt['keyword_list'].map(preprocess)
    gen_corpus(processed_txt,MODEL)
    
    lda_model=LDA_model(NUM_TOPICS)

    write_topics(lda_model)
    
    return

def topic_index(resultsjson):
    ###########TOPIC INDEXING#########################
    bow_corpus = corpora.MmCorpus(BOW_CORPUS_PATH)
    #dictionary = corpora.Dictionary.load(DICT_PATH)
    lda_model_tfidf = gensim.models.LdaModel.load(MODEL_PATH)
    lda_dict = corpora.Dictionary.load(MODEL_PATH+'.id2word')
    print("model loaded successfully")
    while True:
        # go get LIMIT number of items (will duplicate initial select)
        print("about to SQL:")
        resultsjson = selectSQL()
        print("got results, count is: ",len(resultsjson))
        if len(resultsjson) == 0:
            break

        write_imagetopics(resultsjson,lda_model_tfidf,lda_dict)
        print("updated cells")
    print("DONE")

    return
    
def main():
    global MODE
    OPTION, MODE = pick(options, title)

    start = time.time()
    # create_my_engine(db)
    resultsjson = selectSQL()
    print("got results, count is: ",len(resultsjson))

    if MODE==0:topic_model(resultsjson)
    elif MODE==1:topic_index(resultsjson)
    elif MODE==2:calc_optimum_topics(resultsjson)
    end = time.time()
    print (end - start)
    return True

if __name__ == '__main__':
    main()




