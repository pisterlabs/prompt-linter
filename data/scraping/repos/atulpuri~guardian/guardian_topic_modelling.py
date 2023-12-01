# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 23:05:27 2018

@author: Atul
"""

#import os
#os.chdir("D:/Projects/guardian")
import pandas as pd
from elasticsearch import Elasticsearch
import dataPrep
from gensim import corpora
import numpy as np
from gensim.models import TfidfModel
from gensim.models import LsiModel
from gensim.models import ldamodel as lda
from gensim.models.coherencemodel import CoherenceModel
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pyLDAvis
from pyLDAvis import gensim as ldavis
import itertools
import pprint
import json
import logging
logging.basicConfig(filename='lda_model.log', format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

es = Elasticsearch(['localhost'],port=9200) 

search_res = es.search(index = 'guardian', size=2500,
                       timeout='150s',
                       body={"query": { "bool":{
                               "must":{
                               "match": {"sectionName": "World"}}}
                        }}, scroll = '5m')

sid = search_res['_scroll_id']
scroll_size = search_res['hits']['total']

results = []
# Start scrolling
while (scroll_size > 0):
    search_res = es.scroll(scroll_id = sid, scroll = '5m')
    # Update the scroll ID
    sid = search_res['_scroll_id']
    # Get the number of results that we returned in the last scroll
    scroll_size = len(search_res['hits']['hits'])
    print("scrolling...")
    # Do something with the obtained page
    results.extend([[hit['_source']['fields']['bodyText'], hit['_source']['fields']['webPublishedDate']]\
                    for hit in search_res['hits']['hits']])


#search_res['hits']['hits'][0]['_source']['fields']['bodyText']

articles = pd.DataFrame(columns=['articles', 'date'])
articles.articles = results
#[hit['_source']['fields']['bodyText'] for hit in search_res['hits']['hits']]
#del search_res

articles = dataPrep.clean_text(articles, col='articles', remove_unusual=False,
                                    remove_stopwords=True,
                                    remove_numbers=True, stem_words=False,
                                    lemmatize=True, nGram=True)

# input
texts = list(articles.articles)

#with open('clean_world_articles.json','w') as file:
#    json.dump(texts, file)

with open('clean_world_articles.json','r') as file:
    texts = json.load(file)

# turn our tokenized documents into a id <-> term dictionary
dictionary = corpora.Dictionary(texts)
# Filter out words that occur less than 20 documents, or more than 50% of the documents.
dictionary.filter_extremes(no_below=10, no_above=0.5)
# export dictionary
#dictionary.save('dictionary_world_words.dict')
# import dictionary
dictionary = corpora.Dictionary.load('dictionary_world_words.dict')


# convert tokenized documents into a document-term matrix
corpus = [dictionary.doc2bow(text) for text in texts]
# export corpus
#corpora.MmCorpus.serialize('world_articles_corpus.mm', corpus)
# import corpus
corpus = corpora.MmCorpus('world_articles_corpus.mm')


""" finding the optimal number of topics """

tfidf = TfidfModel(corpus, normalize=True)
corpus_tfidf = tfidf[corpus]

# project to 2 dimensions for visualization
lsi = LsiModel(corpus_tfidf, id2word=dictionary, num_topics=2)

# write out coordinates to file
fcoords = open(os.path.join(os.getcwd(), "coords.csv"), 'w')
for vector in lsi[corpus]:
    if len(vector) != 2:
        continue
    fcoords.write("%6.4f\t%6.4f\n" % (vector[0][1], vector[1][1]))
fcoords.close()


MAX_K = 50

X = np.loadtxt(os.path.join(os.getcwd(), "coords.csv"), delimiter="\t")
ks = range(1, MAX_K + 1)

inertias = np.zeros(MAX_K)
diff = np.zeros(MAX_K)
diff2 = np.zeros(MAX_K)
diff3 = np.zeros(MAX_K)

for k in ks:
    print(k)
    kmeans = KMeans(k).fit(X)
    inertias[k - 1] = kmeans.inertia_
    # first difference    
    if k > 1:
        diff[k - 1] = inertias[k - 1] - inertias[k - 2]
    # second difference
    if k > 2:
        diff2[k - 1] = diff[k - 1] - diff[k - 2]
    # third difference
    if k > 3:
        diff3[k - 1] = diff2[k - 1] - diff2[k - 2]
elbow = np.argmin(diff3[3:]) + 3

plt.plot(ks[4:], inertias[4:], "b*-")
#plt.plot(ks[elbow], inertias[elbow], marker='o', markersize=12,
         #markeredgewidth=2, markeredgecolor='r', markerfacecolor=None)
plt.ylabel("Inertia")
plt.xlabel("K")
plt.show()


""" LDA """
# generate LDA model
ldamodel = lda.LdaModel(corpus, num_topics=50, id2word = dictionary, #alpha='auto',
                        passes=20, iterations=400, random_state=1)

fname = "topics50world"
ldamodel.save(fname)
#ldamodel = lda.LdaModel.load(fname, mmap='r')


pprint.pprint(ldamodel.show_topics(num_topics=50,num_words=20))
#pprint.pprint(ldamodel.top_topics(corpus,num_words=10))

ldatopics = [[word for word, prob in topic] for topicid, topic in ldamodel.show_topics(formatted=False)]
lda_coherence = CoherenceModel(topics=ldatopics, texts=texts, dictionary=dictionary,
                               window_size=5, coherence='c_v')
print(lda_coherence.get_coherence())
print(lda_coherence.get_coherence_per_topic())


vis_data = ldavis.prepare(ldamodel, corpus, dictionary)
#pyLDAvis.display(vis_data)
pyLDAvis.save_html(vis_data, 'world_lda50.html')


""""
hey, do you want to play a game?
oh come on!
let's play
"""

# select top n words for each of the LDA topics
top_words = [[word for word, _ in ldamodel.show_topic(topicno, topn=10)] for topicno in range(ldamodel.num_topics)]

# get all top words in all topics, as one large set
all_words = set(itertools.chain.from_iterable(top_words))
print("Can you spot the misplaced word in each topic?")

# for each topic, replace a word at a different index, to make it more interesting
replace_index = np.random.randint(0, 10, ldamodel.num_topics)

replacements = []
for topicno, words in enumerate(top_words):
    other_words = all_words.difference(words)
    replacement = np.random.choice(list(other_words))
    replacements.append((words[replace_index[topicno]], replacement))
    words[replace_index[topicno]] = replacement
    print("%i: %s" % (topicno, ' '.join(words[:10])))

pd.DataFrame([x for x in top_words],index=['topic_{}'.format(i) for i in range(1,16)]).\
            T.to_csv('topics_scrambled.csv', index=False)


print("Actual replacements were:")
print(list(enumerate(replacements)))
