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

logging.basicConfig(filename='/home/norberteke/PycharmProjects/Thesis/logs/SO_past_grid_search_2000.log',
                    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
config_path = "/home/norberteke/PycharmProjects/Thesis/data/SO_past_grid_search_config_log.csv"

#with open(config_path, 'a') as f:
#    writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#    writer.writerow(["Number_of_topics", "Beta", "c_v", "u_mass", "c_uci", "c_npmi"])

def get_data():
    engine = create_engine("mysql+pymysql://norberteke:Eepaiz3h@localhost/norberteke")
    conn = engine.connect()

    activity_SQL = "SELECT full_activity FROM SO_past_activity"
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

topic_num = []
for num in range(3, 50):
    topic_num.append(num)

beta = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15,
        0.16, 0.17, 0.18, 0.19, 0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.30, 0.31, 
        0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.40, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 
        0.48, 0.49, 0.50, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.60, 0.61, 0.62, 0.63, 
        0.64, 0.65, 0.66, 0.67, 0.68, 0.69, 0.70, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 
        0.80, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 
        0.96, 0.97, 0.98, 0.99, 1.00]

corpus = get_data()
dictionary = Dictionary.load(
    "/home/norberteke/PycharmProjects/Thesis/data/SO_past_full_activity_gensimDictionary.dict")

for topic in topic_num:
    print("--------------- Progress: topic num = ", str(topic), " ---------------")
    for b in beta:
        #lda, doc_vecs = fit_model(corpus, dictionary, topic, b)
        lda = fit_model(corpus, dictionary, topic, b)

        temp_file = datapath("/home/norberteke/PycharmProjects/Thesis/grid_search_models/SO_past/topic_" +
                             str(topic) + "_beta_" + str(b) + ".lda")
        lda.save(temp_file)
        #np.savetxt("/home/norberteke/PycharmProjects/Thesis/grid_search_models/SO_past/topic_" +
        #                     str(topic) + "_beta_" + str(b) + "_docVecs.csv", doc_vecs, delimiter=",")
