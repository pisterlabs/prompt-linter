# Author: Olivier Grisel <olivier.grisel@ensta.org>
#         Lars Buitinck
#         Chyi-Kwei Yau <chyikwei.yau@gmail.com>
# License: BSD 3 clause
import os
import re
from tabulate import tabulate
from operator import itemgetter
from time import time
import matplotlib.pyplot as plt
import operator
import numpy as np
import pandas
import pandas as pd
import pyLDAvis as pyLDAvis
#import pyLDAvis.gensim
#from IPython.display import display
import warnings
from collections import OrderedDict
warnings.filterwarnings("ignore", category=DeprecationWarning)
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from gensim.models import CoherenceModel, Nmf, Phrases
from gensim.models.phrases import Phraser
from gensim.corpora import Dictionary


from tf_idf_sci import build_corpus, FIELDS, STEMDICT, write_cluster_to_json, plot_sil

n_samples = 2000
n_features = 100000
n_components = 5
n_top_words = 20
batch_size = 128
init = "nndsvda"

results_umass = {}
results_cv = {}

def get_people_for_topic(people, series):
    people_per_topic = {}
    for i in range(0, len(series)):
        people_per_topic[''.join([j for j in people[i] if not j.isdigit()]).strip()] = str(series[i])

    return people_per_topic


def plt_coherence_score(topic_nums, coherence_scores, file_path):
    scores = list(zip(topic_nums, coherence_scores))
    best_num_topics = sorted(scores, key=itemgetter(1), reverse=True)[0][0]

    # Plot the results
    fig = plt.figure(figsize=(15, 7))

    plt.plot(
        topic_nums,
        coherence_scores,
        linewidth=3,
        color='#4287f5'
    )

    plt.xlabel("Topic Num", fontsize=14)
    plt.ylabel("Coherence Score", fontsize=14)
    plt.title('Coherence Score by Topic Number - Best Number of Topics: {}'.format(best_num_topics), fontsize=18)
    plt.xticks(np.arange(5, max(topic_nums) + 1, 5), fontsize=12)
    plt.yticks(fontsize=12)

    file_name = 'c_score'

    fig.savefig(
        file_path + file_name + '.png',
        dpi=fig.dpi,
        bbox_inches='tight'
    )

    plt.show()
    return 0


def print_dict(results_dict, model_name, model_desc, reverse):
    i = 0;

    for key in sorted(results_dict, reverse=reverse):
        i = i + 1
        dict_fields = results_dict[key].split("\n", 1)
        print("CoherenceModel: {} ({}), Model Rank: {}, Score {}, Fields: {} ".format(model_name, model_desc, str(i), str(key) ,dict_fields[0]))
        print("{}\n".format(dict_fields[1]).replace("0", "Group", 1).replace("1", "Terms", 1))

def process_field(fields):

    def disp_topics(nmf, model, model_name):

        score = model.get_coherence()
        # print("Score: {}".format(score))
        # print("Topics: {}\n\n".format(nmf.show_topics(num_words=20)))
        tf = pd.DataFrame(nmf.show_topics(num_words=20))
        length = len(tf[1].values)
        for i in range(length):
            tf[1][i] = re.sub('[0-9\.\*]*', '', tf[1][i])
        tf.rename(columns={tf.columns[0]: 'group'})
        tf.rename(columns={tf.columns[1]: 'terms'})
        text_result = (tabulate(tf, showindex=False, headers=tf.columns))

        if model_name == "u_mass":  # sort by higher is better
            results_umass[score] = " ".join(fields) + "\n" + text_result
        if model_name == "c_v":  # sort by lower is better
            results_cv[score] = " ".join(fields) + "\n" + text_result



    #print("Loading dataset...")
    #t0 = time()
    abstracts_df = build_corpus(fields, do_stemming=True, do_remove_common=True)
    data = abstracts_df["text"].tolist()
    people = abstracts_df["people"].tolist()

    def convert(lst):
        return ''.join(lst).split()

    data_samples = data[:n_samples]
    texts = [convert(item) for item in data_samples]

    bigram = Phrases(texts, min_count=5, threshold=10)  # higher threshold fewer phrases.
    trigram = Phrases(bigram[texts], threshold=10)
    quadgram = Phrases(trigram[texts], threshold=10)
    bigram_mod = Phraser(bigram)
    trigram_mod = Phraser(trigram)
    quadgram_mod = Phraser(quadgram)


    bigrams =  [bigram_mod[doc] for doc in texts]
    trigrams = [trigram_mod[bigram_mod[doc]] for doc in texts]
    quadgrams = [quadgram_mod[trigram_mod[bigram_mod[doc]]] for doc in texts]
    texts = quadgrams

    dictionary = Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    #print("done in %0.3fs." % (time() - t0))

    #print("Running nmf model...")
    t0 = time()
    nmf = Nmf(corpus=corpus, id2word=dictionary, num_topics=5)

    #print("done in %0.3fs." % (time() - t0))
    t0 = time()
    #print("Running coherence model (u_mass)...")

    cm = CoherenceModel(model=nmf, corpus=corpus, dictionary=dictionary, coherence='u_mass')
    disp_topics(nmf, cm, "u_mass");
    cm = CoherenceModel(model=nmf, texts=texts, dictionary=dictionary, coherence='c_v')
    disp_topics(nmf, cm, "c_v");


    #print("done in %0.3fs." % (time() - t0))
    #print(cm)
    #prepared = pyLDAvis.gensim.prepare(nmf, corpus, dictionary)
    #print(prepared)
 #   display(pyLDAvis.display(prepared))

    # output_dir = "output-te/" + "-".join(fields) + "/"
    # os.makedirs(output_dir, exist_ok=True)




def main():
    for field_set in FIELDS:
        process_field(field_set)





    print_dict(results_umass, "u_mass", "higher is better", True)
    print_dict(results_cv, "c_v", "lower is better", False)


if __name__ == "__main__":
    main()
    quit()
