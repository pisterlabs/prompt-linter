# ##import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from pprint import pprint

#Gensim
# import little_mallet_wrapper as lmw
from gensim.models import CoherenceModel
from gensim.models.ldamulticore import LdaMulticore

#lda vis
import pyLDAvis
import pyLDAvis.gensim_models
import pyLDAvis.gensim_models as gensimvis

import os
from pathlib import Path
import sys
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import pickle 
import time
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
MKL_NUM_THREADS=1 
NUMEXPR_NUM_THREADS=1 
OMP_NUM_THREADS=1

""" Set Mode """
mode = int(sys.argv[1])

""" Functions """
def extract_topics(row):
    # Get main topic in each document
    row.sort(key=itemgetter(1))
    # Get the Dominant topic, Perc Contribution and Keywords for each document
    topic_num, prop_topic = row[-1]
    wp = ldamodel.show_topic(topic_num)
    topic_keywords = ", ".join([word for word, prop in wp])
    sent_topics_df = [int(topic_num), round(prop_topic,4), topic_keywords]
    return(sent_topics_df)

def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3, random_state = 0):
    """
    Compute u_mass coherence for various number of topics
    """

    if len(texts) > 1e6: #model needs enough passes and iterations to converge
        passes = 2
        iterations = 50
        # distributed = True
        # chunksize = 10000
    else:  
        passes = 5
        iterations = 250
        # distributed = False
        # chunksize = 10000
        
    print("Selecting: " + str(passes) + " epochs and " + str(iterations) + " iterations!")
    
    coherence_values = []
    model_list = []
    time_start = time.time()

    #modify for faster runtime (either chunksize, iterations or passes)
    for num_topics in range(start, limit, step):
        print("Extracting: " + str(num_topics) + " topics!")
        model=LdaMulticore(corpus=corpus, id2word=dictionary, num_topics=num_topics, alpha = "asymmetric", 
        chunksize=10000, random_state=random_state, passes = passes, iterations = iterations, workers=4)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, coherence='c_npmi')
        coherence_values.append(coherencemodel.get_coherence())
        time_now = time.time()
        print("Time since start :" + str(round(time_now-time_start, 2)))
    return model_list, coherence_values

""" Import Data """

print("Mode: " + str(mode))
text_types = ["posts", "comments", "all"]
text_type = text_types[mode-1]

#lemma
with open('../data/auxiliary/lemma_' + text_type + '.pkl',  'rb') as f:
    lemma = pickle.load(f)
    
#corpus
with open('../data/auxiliary/corpus_' + text_type + '.pkl',  'rb') as f:
    corpus = pickle.load(f)
    
#dicts
with open('../data/auxiliary/dict_' + text_type + '.pkl',  'rb') as f:
    id2word = pickle.load(f)

print("Load Data")
if mode == 1:
    df = pd.read_csv("../data/auxiliary/all_posts_processed.csv", keep_default_na=False, lineterminator="\n")
    df["text_type"] = text_type
    texts       = df.combined_text.tolist()
    ids = df["id"].tolist()
elif mode == 2:
    df = pd.read_csv("../data/auxiliary/all_comments_processed.csv", keep_default_na=False, lineterminator = "\n")
    df["text_type"] = text_type
    texts    = df.text.tolist()
    ids = df["id"].tolist()
else:
    df_posts = pd.read_csv("../data/auxiliary/all_posts_processed.csv", keep_default_na=False, lineterminator = "\n")
    df_posts["text_type"] = text_type
    df_comments = pd.read_csv("../data/auxiliary/all_comments_processed.csv", keep_default_na=False, lineterminator = "\n")
    df_comments["text_type"] = text_type
    df = pd.concat([df_posts, df_comments], ignore_index=True)
    posts       = df_posts.combined_text
    comments    = df_comments.text
    texts = posts.tolist() + comments.tolist()
    ids = df["id"].tolist()

""" Model Evaluation/Comparison """
Path("../data/images").mkdir(exist_ok=True, parents=True)
Path("../data/auxiliary/pylda").mkdir(exist_ok=True, parents=True)
Path("../data/results/topics").mkdir(exist_ok=True, parents=True)

if not os.path.isfile("../data/results/topics/coherence_values_" + text_type + ".pkl"):
    model_list, coherence_values = compute_coherence_values(dictionary=id2word,
                                                            corpus=corpus, 
                                                            texts=lemma,
                                                            start=2, 
                                                            limit=31, 
                                                            step=4, 
                                                            random_state = 0)

    #save best model
    print("Finished fitting models!")
    print("Save best model and coherence scores")
    # idx = np.argmin([abs(x) for x in coherence_values])
    idx = np.argmax(coherence_values)
    opt_model = model_list[idx]

    with open("../data/results/topics/optimal_model_" + text_type + ".pkl", "wb") as f:
        pickle.dump(opt_model, f)

    # Show graph
    limit=31; start=2; step=4
    x = range(start, limit, step)
    opt_nr = x[idx]
    plt.plot(x, coherence_values)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    plt.savefig('../data/images/coherence_values_' + text_type + '.png')

    with open("../data/results/topics/coherence_values_" + text_type + ".pkl", "wb") as f:
        pickle.dump(coherence_values, f)
        
else:
    print("Load saved model and coherence scores")
    with open("../data/results/topics/optimal_model_" + text_type + ".pkl", "rb") as f:
           opt_model = pickle.load(f)

    with open("../data/results/topics/coherence_values_" + text_type + ".pkl", "rb") as f:
            coherence_values = pickle.load(f)
    
""" Visualize the topics (interactive) """
file_paths = ['../data/auxiliary/pylda/ldavis_posts_opt', 
              '../data/auxiliary/pylda/ldavis_comments_opt',
              '../data/auxiliary/pylda/ldavis_all_opt']

#posts
# pyLDAvis.enable_notebook()
LDAvis_data_filepath = os.path.join(file_paths[mode-1])
if not os.path.isfile(LDAvis_data_filepath + ".html"):
    print("Create pyLDA plots")
    LDAvis_prepared = gensimvis.prepare(opt_model, corpus, id2word, mds='mmds')
    with open(LDAvis_data_filepath, 'wb') as f:
        pickle.dump(LDAvis_prepared, f)
    # load the pre-prepared pyLDAvis data from disk
    with open(LDAvis_data_filepath, 'rb') as f:
        LDAvis_prepared = pickle.load(f)
    pyLDAvis.save_html(LDAvis_prepared, LDAvis_data_filepath + '.html')

file_paths = ['../data/auxiliary/df_topic_sent_', 
              '../data/results/topics/topic_distribution_',
              '../data/results/topics/typical_docs_']

file_paths = [file + text_type + "_" + str(opt_nr) + ".csv" for file in file_paths]

print("Get topics for each doc")
if not os.path.isfile(file_paths[0]):
    print("Extract Meta-information")
    #apply model on texts, adjust based on best score
    df_topic_sents_keywords = pd.DataFrame(topics_sentences, columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords'])
    df_topic_sents_keywords["Text"] = texts
    df_topic_sents_keywords["id"] = ids
    df_topic_sents_keywords.to_csv("../data/auxiliary/df_sent_" + text_type + "_" + str(opt_nr) + ".csv", index=False)

    # Format
    df_dominant_topic = df_topic_sents_keywords.reset_index()
    df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text', 'id']
    
    #save new dataframe:
    df_dominant_topic.to_csv("../data/results/topics/topics_" + text_type + "_" + str(opt_nr) + ".csv", index = False)

else:
    print("Load Meta-information")
    df_topic_sents_keywords = pd.read_csv("../data/auxiliary/df_sent_" + text_type + "_" + str(opt_nr) + ".csv", lineterminator="\n")

    # Format
    df_dominant_topic = df_topic_sents_keywords.reset_index()
    df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text', 'id']

    #save new dataframe:
    df_dominant_topic.to_csv("../data/results/topics/topics_" + text_type + "_" + str(opt_nr) + ".csv", index = False)
    
""" Meta-analysis - Topic distributions """

if not os.path.isfile(file_paths[1]):

    print("Get topic distribution")
    topic_counts = df_topic_sents_keywords['Dominant_Topic'].value_counts()
    # Percentage of Documents for Each Topic
    topic_contribution = round(topic_counts/topic_counts.sum(), 4)
    # Topic Number and Keywords
    topic_num_keywords = df_topic_sents_keywords[['Dominant_Topic', 'Topic_Keywords']].drop_duplicates().reset_index(drop=True)
    # Concatenate Column wise
    df_dominant_topics = pd.concat([topic_num_keywords, topic_counts, topic_contribution], axis=1)
    # Change Column names
    df_dominant_topics.columns = ['Dominant_Topic', 'Topic_Keywords', 'Num_Documents', 'Perc_Documents']

    # Save file
    df_dominant_topics.to_csv("../data/results/topics/topic_distribution_" + text_type + "_" + str(opt_nr) + ".csv", index = False)

else:
    pass

#Find most representative docs
if not os.path.isfile(file_paths[2]):
    print("Get typical docs for each topic")
    sent_topics_sorteddf = pd.DataFrame()
    sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')

    for i, grp in sent_topics_outdf_grpd:
        sent_topics_sorteddf = pd.concat([sent_topics_sorteddf,
                                                 grp.sort_values(['Perc_Contribution'], ascending=[0]).head(1)],
                                                axis=0)
    # Reset Index
    sent_topics_sorteddf.reset_index(drop=True, inplace=True)
    # Format
    sent_topics_sorteddf.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Text", 'id']

    # Save file
    sent_topics_sorteddf.to_csv("../data/results/topics/typical_docs_" + text_type + "_" + str(opt_nr) + ".csv", index = False)
else:
    pass



















