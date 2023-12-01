# -*- coding: utf-8 -*-
# # Tests of topic modelling with gensim

# ### Import data

# + {"active": ""}
# Ressources:
#

# +
import os
import json
from nltk import word_tokenize
from stop_words import get_stop_words
import nltk
from nltk.corpus import stopwords

import gensim
from gensim.models import CoherenceModel
from gensim import corpora
from gensim.utils import simple_preprocess

import pickle

import pyLDAvis
import pyLDAvis.gensim  
import matplotlib.pyplot as plt
# %matplotlib inline

from nltk.stem.snowball import FrenchStemmer

import pandas as pd

stemmer = FrenchStemmer()
# -


# ## Work on several articles

# +
# Choose the parametres

nb_articles = 20000
words_no_above = 0.5
NUM_TOPICS = 20



# +
articles = []
titles = []
print('...')
for year in os.listdir('./cleaned_articles/'):
    if year != '.DS_Store':
        for filename in os.listdir('./cleaned_articles/'+year):
            if (len(articles)<nb_articles):
                    path = './cleaned_articles/'+year+'/'+filename
                    file = open(path, 'r')
                    read_file = json.loads(file.read())[0]
                    titles.append(read_file['title'])
                    read_file = read_file['text']

                    #removing punctuation, stop words and lemmatize
                    text = "".join([char if (char.isalnum() or char==" ") else " " for char in read_file])
                    text = word_tokenize(text)
                    text = [word.lower() for word in text]
                    text = [stemmer.stem(word) for word in text]
                    text = [word for word in text if len(word)>3]
                    articles.append(text)
    
dictionary2 = corpora.Dictionary(articles)
dictionary2.filter_extremes(no_above=words_no_above)
corpus2 = [dictionary2.doc2bow(article) for article in articles]
pickle.dump(corpus2, open('corpus2.pkl', 'wb'))
dictionary2.save('dictionary2.gensim')
print('Done')
# -


# ### Create différent topics with gensim

# +
# Create the topics
import numpy as np


# function to enter the topic in eta
def add_topic_to_eta(words, topic_nb, eta):
    print('Topic number '+str(topic_nb))
    for word in words:
        if word in dictionary2.token2id:
            eta[topic_nb, dictionary2.token2id[word]]*=10
        else:
            print(word,' not in the dictionary')

def create_eta(topics):
    eta = np.ones((NUM_TOPICS, len(dictionary2)))*1/NUM_TOPICS
    if NUM_TOPICS < len(topics):
        print('You try to seed to many topics')
    else:
        topic_nb = 0
        for topic in topics:
            add_topic_to_eta(topic, topic_nb, eta)
            topic_nb += 1
    return eta  


# +
# Create topics (especially little topics)

# global topicq
words_war = ['guerr', 'etat', 'polit', 'président', 'arme', 'mort', 'conflit']
words_intern_politics = ['ministr', 'président', 'polit', 'franc', 'gouvern', 'municipal', 'sénat', 'élect', 'élir']
words_intern_economics = ['impôt', 'fiscal', 'smic', 'emplois', 'inflat', 'revenus']
words_economics = ['économ', 'taux', 'euro', 'bour', 'dollar', 'baiss', 'croissanc', 'boursi', 'inflat', 'milliard', 'banqu']

# special topics
words_sport = ['sport', 'olymp', 'athlet', 'défait', 'victoir', 'match', 'foot', 'champion', 'football', 'basket', 'roland', 'médaill', 'club', 'jeux', 'supporter'] # marche, associé à la tété chaine, cavle etc ...
words_ecology = ['durabl', 'planet', 'vert', 'ecolo', 'carbon', 'climat', 'écolog', 'énerg', 'nucléair', 'serr', 'gaz', 'réchauff']
words_school = ['scolair','écol','enseignement', 'enseign','étud','lycéen','orient','enfant', 'réform']
words_religion= ['églis', 'eglis', 'prêtr', 'vatican', 'relig', 'mess', 'évêqu', 'chrétien', 'catholic', 'catholique', 'cathol', 'abbé']
words_space = ['espac', 'astronaut', 'navet', 'spatial', 'lunair', 'satellit', 'apollo', 'terr', 'solair', 'orbit', 'cabin', 'oxygen']
words_shoah = ['juif', 'extermin', 'auschwitz', 'antisémit', 'camp', 'hitler', 'nazism', 'adolf', 'ghetto', 'rabbin', 'wagon', 'reich']
words_music = ['musiqu', 'concert', 'chanson', 'album']
words_seism = ['séism', 'humanitair', 'mort', 'trembl', 'terr']
# film ? télé ? reseaux sociaux ?



#words_greve = ['emplois', 'chômag', 'grev', 'social', 'manifest']


topics = [words_ecology, words_sport, words_school, words_religion, words_space, words_shoah, words_music, words_seism]


# +
#dictionary2.dfs[dictionary2.token2id['sport']]
# -

eta = create_eta(topics)

# +
print('...')

ldamodel2 = gensim.models.ldamodel.LdaModel(
    corpus2, num_topics = NUM_TOPICS, id2word=dictionary2, passes=10, per_word_topics=True, update_every=1, iterations=50, eta=eta
)
ldamodel2.save('model2.gensim')

ldamodel2 = gensim.models.ldamodel.LdaModel.load('model2.gensim')

topics2 = ldamodel2.print_topics(num_words=10)


for topic in topics2:
    print(topic)
    print()
# -
# ### Plot topics

# Visualize the topics
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(ldamodel2, corpus2, dictionary2)
vis


# ## Print most relevent document (not working)

def format_topics_sentences(ldamodel, corpus, texts):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        print(i)
        print(row)
        row.sort(key=(lambda x: x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)


# +
df_topic_sents_keywords = format_topics_sentences(ldamodel=ldamodel2, corpus=corpus2, texts=titles)
# Group top 5 sentences under each topic
sent_topics_sorteddf_mallet = pd.DataFrame()

sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')

for i, grp in sent_topics_outdf_grpd:
    sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet, 
                                             grp.sort_values(['Perc_Contribution'], ascending=[0]).head(1)], 
                                            axis=0)

# Reset Index    
sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)

# Format
sent_topics_sorteddf_mallet.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Text"]

# Show
sent_topics_sorteddf_mallet.head()
# -


