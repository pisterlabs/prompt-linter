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

logging.basicConfig(filename='/home/norberteke/PycharmProjects/Thesis/logs/GH_recent_grid_search_3000.log',
                    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
config_path = "/home/norberteke/PycharmProjects/Thesis/data/GH_recent_grid_search_config_log.csv"

#with open(config_path, 'a') as f:
#    writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#    writer.writerow(["Number_of_topics", "Beta", "c_v", "u_mass", "c_uci", "c_npmi"])

def get_data():
    engine = create_engine("mysql+pymysql://norberteke:Eepaiz3h@localhost/norberteke")
    conn = engine.connect()

    activity_SQL = "SELECT full_activity FROM GH_recent_activity"
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

def get_data_from_file(path):
    with open(path) as f: texts = [line.rstrip('\n') for line in f]
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

topic_num = []
for val in range(60, 150):
  topic_num.append(val)
  
beta = [0.001, 0.003, 0.005, 0.007, 0.009, 0.01, 0.03, 0.05, 0.07, 0.09, 
      0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

corpus = get_data()
dictionary = Dictionary.load(
    "/home/norberteke/PycharmProjects/Thesis/data/GH_recent_full_activity_gensimDictionary.dict")

for topic in topic_num:
    print("--------------- Progress: topic num = ", str(topic), " ---------------")
    for b in beta:
        #lda, doc_vecs = fit_model(corpus, dictionary, topic, b)
        lda = fit_model(corpus, dictionary, topic, b)

        temp_file = datapath("/home/norberteke/PycharmProjects/Thesis/grid_search_models/GH_recent/topic_" +
                             str(topic) + "_beta_" + str(b) + ".lda")
        lda.save(temp_file)
        #np.savetxt("/home/norberteke/PycharmProjects/Thesis/grid_search_models/GH_recent/topic_" +
        #                     str(topic) + "_beta_" + str(b) + "_docVecs.csv", doc_vecs, delimiter=",")
