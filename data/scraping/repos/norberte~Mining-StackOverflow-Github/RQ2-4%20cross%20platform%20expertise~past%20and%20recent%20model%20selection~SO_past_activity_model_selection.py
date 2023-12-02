from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
from gensim.sklearn_api import LdaTransformer
from skopt import dump
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.callbacks import CheckpointSaver
from skopt.utils import use_named_args
import csv
import logging
from sqlalchemy import create_engine
from sqlalchemy.sql import text
import pandas as pd
from skopt import load

logging.basicConfig(filename='/home/norberteke/PycharmProjects/Thesis/logs/SO_past_optimizer_10000.log', format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
config_path = "/home/norberteke/PycharmProjects/Thesis/data/SO_past_model_parameter_config_log.csv"
runs = 8000

def saveModelConfigs(model, coherence, u_mass, c_uci, c_npmi, path):
    with open(path, 'a') as f:
        writer = csv.writer(f, delimiter = ',', quotechar='"', quoting = csv.QUOTE_MINIMAL)
        writer.writerow([str(model.num_topics), str(model.eta), str(coherence), str(u_mass), str(c_uci), str(c_npmi)])


def evaluateModel(model):
    cm = CoherenceModel(model=model, texts=texts, coherence='c_v')
    coherence = cm.get_coherence()  # get coherence value
    return coherence


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

#with open("/home/norberteke/PycharmProjects/Thesis/data/SO_past_full_activity_corpus.txt", 'r') as f:
#    texts = f.read().splitlines()

dictionary = Dictionary.load("/home/norberteke/PycharmProjects/Thesis/data/SO_past_full_activity_gensimDictionary.dict")

corpus = [dictionary.doc2bow(text) for text in texts]
# output_fname = get_tmpfile("/home/norberteke/PycharmProjects/Thesis/data/SO_recent_full_activity_gensimCorpus.mm")
# MmCorpus.serialize(output_fname, corpus)

model = LdaTransformer(id2word=dictionary, alpha='auto', iterations=100, random_state=2019)

# The list of hyper-parameters we want to optimize. For each one we define the bounds,
# the corresponding scikit-learn parameter name
space = [Integer(20, 500, name='num_topics'), Real(0.001, 200, name='eta')]

# this decorator allows your objective function to receive a the parameters as keyword arguments.
# This is particularly convenient when you want to set scikit-learn estimator parameters
@use_named_args(space)
def objective(**params):
    model.set_params(**params)
    lda = model.fit(corpus)
    coherence = evaluateModel(lda.gensim_model)

    try:
        cm = CoherenceModel(model=lda.gensim_model, corpus=corpus, dictionary=dictionary, coherence='u_mass')
        u_mass = cm.get_coherence()

        cm = CoherenceModel(model=lda.gensim_model, texts=texts, coherence='c_uci')
        c_uci = cm.get_coherence()

        cm = CoherenceModel(model=lda.gensim_model, texts=texts, coherence='c_npmi')
        c_npmi = cm.get_coherence()

        saveModelConfigs(lda, coherence, u_mass, c_uci, c_npmi, config_path)
    except:
        saveModelConfigs(lda, coherence, "Invalid", "Invalid", "Invalid", config_path)

    return 1 - coherence  # to maximize coherence score, minimize 1 - coherence score


checkpoint_saver = CheckpointSaver("/home/norberteke/PycharmProjects/Thesis/data/new_SO_past_param_optimizer_checkpoint.pkl")
#res = load('/home/norberteke/PycharmProjects/Thesis/data/new_SO_past_param_optimizer_checkpoint.pkl')
#x0 = res.x_iters
#y0 = res.func_vals

#res_gp = gp_minimize(func=objective, dimensions=space, x0=x0, y0=y0, n_calls=100, n_restarts_optimizer= 5,
#                    n_random_starts=5, callback=[checkpoint_saver], random_state=2019, verbose=True)
res_gp = gp_minimize(func=objective, dimensions=space,  n_calls=100, n_restarts_optimizer= 5,
                    n_random_starts=5, callback=[checkpoint_saver], random_state=2019, verbose=True)

try:
    dump(res_gp, '/home/norberteke/PycharmProjects/Thesis/data/new_SO_past_param_optimizer_result_100.pkl')
except:
    dump(res_gp,
             '/home/norberteke/PycharmProjects/Thesis/data/new_SO_past_param_optimizer_result_without_objective_100.pkl', store_objective=False)
finally:
    try:
        print("Best score=%.6f" % (1 - res_gp.fun))  # to make up for 1 - coherence (reverse the equation)

        print("""Best parameters:
            - num_topics=%d
            - eta=%.6f""" % (res_gp.x[0], res_gp.x[1]))
    except:
        print("Problems occured")
        print(str(res_gp.fun))
        print(str(res_gp.x[0]), str(res_gp.x[1]))
