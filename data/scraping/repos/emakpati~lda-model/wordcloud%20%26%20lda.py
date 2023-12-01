### All Dependencies ###

import pandas as pd
import string
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS as stopwords
import nltk
nltk.download("wordnet")
from nltk.stem import WordNetLemmatizer as lemm, SnowballStemmer as stemm
from nltk.stem.porter import *
import numpy as np
np.random.seed(0)
from gensim import corpora, models
from gensim.models import CoherenceModel
from pprint import pprint
import pyLDAvis.gensim as pyldavis
import pyLDAvis


### Word Cloud ###

df = pd.read_csv(r"/Users/Ekene/Desktop/tweet_data.csv", names=["id", "text", "date", "name",
                                                                 "username", "followers", "loc"])


def clean(txt):
    txt = str(txt).split()
    for item in txt:
        if "http" in item:
            txt.remove(item)
    txt = (" ".join(txt))
    return txt


text = (df.text.apply(clean))


wc = cloud(background_color='white', colormap="tab10").generate(" ".join(text))

plt.axis("off")
plt.text(2, 210, "Generated using word_cloud and this post's dataset.", size = 5, color="grey")

plt.imshow(wc)
plt.show()


### Data ###


df = pd.read_csv(r"/Users/Ekene/Desktop/tweet_data.csv", names= ["id", "text", "date", "name",
                                                                 "username", "followers", "loc"])


### Data Cleaning ###

ppl = ["berniesanders", "kamalaharris", "joebiden", "ewarren"]


def clean(txt):
    txt = str(txt.translate(str.maketrans("", "", string.punctuation))).lower()
    txt = str(txt).split()
    for item in txt:
        if "http" in item:
            txt.remove(item)
        for item in ppl:
            if item in txt:
                txt.remove(item)
    txt = (" ".join(txt))
    return txt


df.text = df.text.apply(clean)



### Data Prep ###

# print(stopwords)

# If you want to add to the stopwords list: stopwords = stopwords.union(set(["add_term_1", "add_term_2"]))



### Lemmatize and Stem ###

stemmer = stemm(language="english")


def lemm_stemm(txt):
    return stemmer.stem(lemm().lemmatize(txt, pos="v"))


def preprocess(txt):
    r = [lemm_stemm(token) for token in simple_preprocess(txt) if       token not in stopwords and len(token) > 2]
    return r


proc_docs = df.text.apply(preprocess)



### The Model ###

dictionary = gensim.corpora.Dictionary(proc_docs)
dictionary.filter_extremes(no_below=5, no_above= .90)
# print(dictionary)

n = 5 # Number of clusters we want to fit our data to
bow = [dictionary.doc2bow(doc) for doc in proc_docs]
lda = gensim.models.LdaMulticore(bow, num_topics= n, id2word=dictionary, passes=2, workers=2)
# print(bow)

for id, topic in lda.print_topics(-1):
    print(f"TOPIC: {id} \n WORDS: {topic}")



### Coherence Scoring ###

coh = CoherenceModel(model=lda, texts= proc_docs, dictionary = dictionary, coherence = "c_v")
coh_lda = coh.get_coherence()
print("Coherence Score:", coh_lda)

lda_display = pyldavis.prepare(lda, bow, dictionary)
pyLDAvis.show(lda_display)