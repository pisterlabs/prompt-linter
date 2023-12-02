from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
from gensim.sklearn_api import LdaTransformer
import csv
import logging
from sqlalchemy import create_engine
from sqlalchemy.sql import text
import pandas as pd
from gensim.test.utils import datapath
import numpy as np

logging.basicConfig(filename='/home/norberteke/PycharmProjects/Thesis/logs/SO_recent_grid_search_2000.log',
                    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
config_path = "/home/norberteke/PycharmProjects/Thesis/data/SO_recent_grid_search_config_log.csv"

with open(config_path, 'a') as f:
    writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(["Number_of_topics", "Beta", "c_v", "u_mass", "c_uci", "c_npmi"])

def get_data():
    engine = create_engine("mysql+pymysql://norberteke:Eepaiz3h@localhost/norberteke")
    conn = engine.connect()

    activity_SQL = "SELECT full_activity FROM SO_recent_activity"
    data = pd.read_sql_query(sql=text(activity_SQL), con=conn)

    engine.dispose()

    texts = []
    for line in data['full_activity']:
        if line is None:
            texts.append([""])
        elif len(line.split()) < 1:
            texts.append([""])
        else:
            texts.append(line.split())
    return texts


def evaluateModel(model, texts):
    cm = CoherenceModel(model=model, texts=texts, coherence='c_v')
    coherence = cm.get_coherence()  # get coherence value
    return coherence


def saveModelConfigs(model, coherence, u_mass, c_uci, c_npmi, path):
    with open(path, 'a') as f:
        writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(
            [str(model.num_topics), str(model.eta), str(coherence), str(u_mass), str(c_uci), str(c_npmi)])


def fit_model(corpora, dictionary, topicNum, beta):
    corpus = [dictionary.doc2bow(text) for text in corpora]

    model = LdaTransformer(id2word=dictionary, num_topics=topicNum, alpha='auto', eta=beta, iterations=100, random_state=2019)
    lda = model.fit(corpus)
    #docvecs = lda.transform(corpus)
    coherence = evaluateModel(lda.gensim_model, corpora)

    try:
        cm = CoherenceModel(model=lda.gensim_model, corpus=corpus, dictionary=dictionary, coherence='u_mass')
        u_mass = cm.get_coherence()

        cm = CoherenceModel(model=lda.gensim_model, texts=corpora, coherence='c_uci')
        c_uci = cm.get_coherence()

        cm = CoherenceModel(model=lda.gensim_model, texts=corpora, coherence='c_npmi')
        c_npmi = cm.get_coherence()

        saveModelConfigs(lda, coherence, u_mass, c_uci, c_npmi, config_path)
    except:
        saveModelConfigs(lda, coherence, "Invalid", "Invalid", "Invalid", config_path)
    #return lda.gensim_model, docvecs
    return lda.gensim_model

topic_num=[]
for num in range(3, 20):
    topic_num.append(num)


beta = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06,0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
        1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5 , 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5,
        14, 14.5, 15,15.5, 16, 16.5, 17, 17.5, 18, 18.5, 19, 19.5, 20, 21, 22, 23, 24, 25, 30, 35, 40, 45, 50]

corpus = get_data()
dictionary = Dictionary.load(
    "/home/norberteke/PycharmProjects/Thesis/data/SO_recent_full_activity_gensimDictionary.dict")

for topic in topic_num:
    print("--------------- Progress: topic num = ", str(topic), " ---------------")
    for b in beta:
        #lda, doc_vecs = fit_model(corpus, dictionary, topic, b)
        lda = fit_model(corpus, dictionary, topic, b)

        temp_file = datapath("/home/norberteke/PycharmProjects/Thesis/grid_search_models/SO_recent/topic_" +
                             str(topic) + "_beta_" + str(b) + ".lda")
        lda.save(temp_file)
        #np.savetxt("/home/norberteke/PycharmProjects/Thesis/grid_search_models/SO_recent/topic_" +
        #                     str(topic) + "_beta_" + str(b) + "_docVecs.csv", doc_vecs, delimiter=",")
