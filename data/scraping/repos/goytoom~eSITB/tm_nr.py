# ##import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Gensim
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
from timeit import default_timer as timer
from operator import itemgetter

""" Set Mode """
mode = int(sys.argv[1])
opt_nr = int(sys.argv[2])

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

""" Import Data """
text_types = ["posts", "comments", "all"]
text_type = text_types[mode-1]

print("Load corpora")
#lemma
with open('../data/auxiliary/lemma_' + text_type + '.pkl',  'rb') as f:
    lemma = pickle.load(f)

#corpus
with open('../data/auxiliary/corpus_' + text_type + '.pkl',  'rb') as f:
    corpus = pickle.load(f)

#dicts
with open('../data/auxiliary/dict_' + text_type + '.pkl',  'rb') as f:
    id2word = pickle.load(f)

print("Load data")
if mode == 1:
    df = pd.read_csv("../data/auxiliary/all_posts_processed.csv", keep_default_na=False, lineterminator="\n")
    df["text_type"] = text_type
    texts       = df.combined_text.tolist() #.sample(100).reset_index(drop=True)
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

if not os.path.isfile("../data/results/topics/ldamodel_" + text_type + "_" + str(opt_nr) + ".pkl"):
    print("Fit Model")
    if mode == 1:
        passes = 5
        iterations = 250
    else:
        passes = 2
        iterations = 50

    ldamodel=LdaMulticore(corpus=corpus, id2word=id2word, num_topics=opt_nr, alpha="asymmetric", chunksize=10000, random_state=0, iterations=iterations, passes=passes, workers=4)
    coherencemodel = CoherenceModel(model=ldamodel, texts=lemma, coherence='c_npmi')
    coherence_score = coherencemodel.get_coherence()

    with open("../data/results/topics/coherence_score_" + text_type + "_" + str(opt_nr) + ".pkl", "wb") as f:
        pickle.dump(coherence_score, f)

    with open("../data/results/topics/ldamodel_" + text_type + "_" + str(opt_nr) + ".pkl", "wb") as f:
        pickle.dump(ldamodel, f)
else:
    with open("../data/results/topics/ldamodel_" + text_type + "_" + str(opt_nr) + ".pkl", "rb") as f:
        ldamodel = pickle.load(f)

    with open("../data/results/topics/coherence_score_" + text_type + "_" + str(opt_nr) + ".pkl", "rb") as f:
        coherence_score = pickle.load(f)

print("Coherence score: " + str(round(coherence_score, 3)))

row_path = "../data/auxiliary/rows_" + text_type + "_" + str(opt_nr) + ".npy"
if not os.path.isfile(row_path):
    arr_raw = ldamodel[corpus]
    arr = np.array(arr_raw, dtype=object)
    np.save(row_path, arr)
else:
    arr = np.load(row_path, allow_pickle=True)

#apply model on texts, adjust based on best score
print("Get topics for each doc")
if not os.path.isfile("../data/auxiliary/df_sent_" + text_type + "_" + str(opt_nr) + ".csv"):
    #extract topics and add remaining information
    topics_sentences = map(extract_topics, arr)
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
    df_topic_sents_keywords = pd.read_csv("../data/auxiliary/df_sent_" + text_type + "_" + str(opt_nr) + ".csv", lineterminator="\n")

    # Format
    df_dominant_topic = df_topic_sents_keywords.reset_index()
    df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text', 'id']

    #save new dataframe:
    df_dominant_topic.to_csv("../data/results/topics/topics_" + text_type + "_" + str(opt_nr) + ".csv", index = False)

""" Meta-analysis - Topic distributions """

# Number of Documents for Each Topic
if not os.path.isfile("../data/results/topics/topic_distribution_" + text_type + "_" + str(opt_nr) + ".csv"):
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
if not os.path.isfile("../data/results/topics/typical_docs_" + text_type + "_" + str(opt_nr) + ".csv"):
    print("Get typical docs")
    sent_topics_sorteddf = pd.DataFrame()
    sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')

    # modify to extent to 10 examples!
    for i, grp in sent_topics_outdf_grpd:
        sent_topics_sorteddf = pd.concat([sent_topics_sorteddf,
                                                 grp.sort_values(['Perc_Contribution'], ascending=[0]).head(10)],
                                                axis=0)
    # Reset Index
    sent_topics_sorteddf.reset_index(drop=True, inplace=True)
    # Format
    sent_topics_sorteddf.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Text", 'id']

    # Save file
    sent_topics_sorteddf.to_csv("../data/results/topics/typical_docs_" + text_type + "_" + str(opt_nr) + ".csv", index = False)
else:
    pass

""" Visualize the topics (interactive) """
file_paths = ['../data/auxiliary/pylda/ldavis_posts_' + str(opt_nr),
              '../data/auxiliary/pylda/ldavis_comments_' + str(opt_nr),
              '../data/auxiliary/pylda/ldavis_all_' + str(opt_nr)]

#posts
# pyLDAvis.enable_notebook()
LDAvis_data_filepath = os.path.join(file_paths[mode-1])
if not os.path.isfile(LDAvis_data_filepath + ".html"):
    print("Create pyLDA")
    LDAvis_prepared = gensimvis.prepare(ldamodel, corpus, id2word, mds='mmds')
    with open(LDAvis_data_filepath, 'wb') as f:
        pickle.dump(LDAvis_prepared, f)
    # load the pre-prepared pyLDAvis data from disk
    with open(LDAvis_data_filepath, 'rb') as f:
        LDAvis_prepared = pickle.load(f)
    pyLDAvis.save_html(LDAvis_prepared, LDAvis_data_filepath + '.html')
