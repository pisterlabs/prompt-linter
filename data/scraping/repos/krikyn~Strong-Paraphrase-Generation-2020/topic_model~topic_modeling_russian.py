import re
from os import system

import numpy as np
import pandas as pd
from pprint import pprint
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt
import logging
import warnings
from lxml import etree

data_words = []

root = etree.parse(r'C:\Users\kiva0319\PycharmProjects\Diploma2020\processed\without_stopwords_short\paraphrases.xml')
root = root.getroot()

count = 0
for element in root[1]:
    print(count)
    count += 1
    element_paragraphs_1 = element[14]
    element_paragraphs_2 = element[15]
    for paragraph in element_paragraphs_1:
        if int(paragraph.attrib.get("words")) >= 5:
            for word in paragraph.text.split(";"):
                if word.isalpha():
                    data_words.append(paragraph.text.split(";"))
        else:
            print("bad paragraph")
    for paragraph in element_paragraphs_2:
        if int(paragraph.attrib.get("words")) >= 5:
            for word in paragraph.text.split(";"):
                if word.isalpha():
                    data_words.append(paragraph.text.split(";"))
        else:
            print("bad paragraph")


bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)

bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

# print(trigram_mod[bigram_mod[data_words[0]]])


def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]


def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]


def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]


data_lemmatized = make_bigrams(data_words)

print(data_lemmatized[:1])

id2word = corpora.Dictionary(data_lemmatized)

texts = data_lemmatized

corpus = [id2word.doc2bow(text) for text in texts]

print(corpus[:1])

lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                            id2word=id2word,
                                            num_topics=20,
                                            random_state=100,
                                            update_every=1,
                                            chunksize=100,
                                            passes=10,
                                            alpha='auto',
                                            per_word_topics=True)

pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]

vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
# vis
# visualisation = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
pyLDAvis.save_html(vis, 'LDA_Visualization.html')
