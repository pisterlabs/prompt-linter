import re
import numpy as np
import pandas as pd
from pprint import pprint

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt
import logging
import warnings
from lxml import etree


def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]


def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)
warnings.filterwarnings("ignore", category=DeprecationWarning)

data_words = []

root = etree.parse(r'C:\Users\kiva0319\PycharmProjects\Diploma2020\processed\paraphrases.xml')
root = root.getroot()

count = 0
bad_paragraphs = 0
for element in root[1]:
    print(count, bad_paragraphs)
    count += 1
    element_paragraphs_1 = element[14]
    element_paragraphs_2 = element[15]
    for paragraph in element_paragraphs_1:
        if int(paragraph.attrib.get("words")) >= 5:
            words = []
            for word in paragraph.text.split(";"):
                if word.isalpha():
                    words.append(word)
            data_words.append(words)
        else:
            print("bad paragraph")
            bad_paragraphs += 1
    for paragraph in element_paragraphs_2:
        if int(paragraph.attrib.get("words")) >= 5:
            words = []
            for word in paragraph.text.split(";"):
                if word.isalpha():
                    words.append(word)
            data_words.append(words)
        else:
            print("bad paragraph")
            bad_paragraphs += 1

bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)

bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

data_lemmatized = make_bigrams(data_words)

print(data_lemmatized[:1])

id2word = corpora.Dictionary(data_lemmatized)

texts = data_lemmatized

corpus = [id2word.doc2bow(text) for text in texts]

print(corpus[:1])

# lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
#                                             id2word=id2word,
#                                             num_topics=10,
#                                             random_state=100,
#                                             update_every=1,
#                                             chunksize=100,
#                                             passes=10,
#                                             alpha='auto',
#                                             per_word_topics=True,
#                                             max_iterations=50)

lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                            id2word=id2word,
                                            num_topics=100,
                                            chunksize=100)

lda_model.save('lda_model_full3')

# pprint(lda_model.print_topics())
# doc_lda = lda_model[corpus]

vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word, mds='mmds')
pyLDAvis.save_html(vis, 'LDA_Visualization.html')
