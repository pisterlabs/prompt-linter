import nltk
nltk.download("stopwords")

import numpy as np
import json
import glob

import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel


import spacy
from nltk.corpus import stopwords

import pandas as pd

import pyLDAvis
import pyLDAvis.gensim


stopwords = stopwords.words('portuguese')


df = pd.read_csv('/content/drive/MyDrive/csvs_analise/oi.csv', on_bad_lines="skip")
#df = pd.read_csv('oi.csv', on_bad_lines='skip')
#df = pd.read_csv("vivo.csv", on_bad_lines="skip")
print(df)

docs = df.Titulos.tolist()

docs

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    nlp = spacy.load('pt_core_news_sm', disable=['parser', 'ner'])
    texts_out = []
    #for text in texts:
       # doc = nlp(" ".join(text))
       # texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])

    #return texts_out

    for text in texts :
      doc = nlp(text)
      new_text = []
      for token in doc:
        if token.pos_ in allowed_postags:
          new_text.append(token.lemma_)
      final = " ".join(new_text)
      texts_out.append(final)
    return texts_out


lenmatized_texts = lemmatization(docs)

lenmatized_texts

def gen_words(texts):
    final = []
    for text in texts:
        new = gensim.utils.simple_preprocess(text, deacc=True)
        final.append(new)

    return final

data_words = gen_words(lenmatized_texts)



bigram_phrases = gensim.models.Phrases(data_words, min_count=5, threshold=10)
trigram_phrases = gensim.models.Phrases(bigram_phrases[data_words], threshold=10)

bigram = gensim.models.phrases.Phraser(bigram_phrases)
trigram = gensim.models.phrases.Phraser(trigram_phrases)

def make_bigrams(texts):
    return([bigram[doc] for doc in texts])

def make_trigrams(texts):
    return ([trigram[bigram[doc]] for doc in texts])

data_bigrams = make_bigrams(data_words)
data_bigrams_trigrams = make_trigrams(data_bigrams)

data_bigrams_trigrams

id2word = corpora.Dictionary(data_words)

corpus = []

for text in data_words:
    new = id2word.doc2bow(text)
    corpus.append(new)

from gensim.models import TfidfModel

id2word = corpora.Dictionary(data_bigrams_trigrams)

texts = data_bigrams_trigrams

corpus = [id2word.doc2bow(text) for text in texts]
# print (corpus[0][0:20])

tfidf = TfidfModel(corpus, id2word=id2word)

low_value = 0.03
words  = []
words_missing_in_tfidf = []
for i in range(0, len(corpus)):
    bow = corpus[i]
    low_value_words = [] #reinitialize to be safe. You can skip this.
    tfidf_ids = [id for id, value in tfidf[bow]]
    bow_ids = [id for id, value in bow]
    low_value_words = [id for id, value in tfidf[bow] if value < low_value]
    drops = low_value_words+words_missing_in_tfidf
    for item in drops:
        words.append(id2word[item])
    words_missing_in_tfidf = [id for id in bow_ids if id not in tfidf_ids] # The words with tf-idf socre 0 will be missing

    new_bow = [b for b in bow if b[0] not in low_value_words and b[0] not in words_missing_in_tfidf]
    corpus[i] = new_bow

lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                            id2word = id2word,
                                            num_topics=20,
                                            random_state=100,
                                            update_every=1,
                                            chunksize=100,
                                            passes=10,
                                            alpha='auto',)

#lda_model.save("vivo.model")

new_model = gensim.models.ldamodel.LdaModel.load("tim.model")

pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word, mds="mmds", R=30)
vis