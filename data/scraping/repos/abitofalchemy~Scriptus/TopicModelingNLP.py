
# coding: utf-8
# https://ahmedbesbes.com/how-to-mine-newsfeed-data-and-extract-interactive-insights-in-python.html
# https://github.com/BitCurator/bitcurator-nlp-gentm
#
# Document Summarization

import json
import pandas as pd

# Getting the data
json_data_file = "/Users/sal.aguinaga/Dropbox_Kyndi/dataSets/Shell/fulltext/get_indexed_fulltext.json"
raw_text = list(filter(lambda l: len(l)>0, 
                             open(json_data_file).read().split("\n")))
docId_json = [json.loads(l) for l in raw_text]

doc_pages_str = []
for k,v in docId_json[4].items():
    if k == 'content':
        cntnt = json.loads(v)
        for p in cntnt['pages']:
            doc_pages_str.append(p['content'])
        
print(len(doc_pages_str))

# doc_pages_str = "".join(doc_pages_str)

# # Stop words
# import nltk; nltk.download('stopwords')
# import packages
import re
import pandas as pd
from pprint import pprint

# Gensim for topic modeling
import gensim
import gensim.corpora as corpora 
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lematization
# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
# python3 -m spacy download en
import spacy
nlp = spacy.load('en', disable=['parser', 'ner'])

# Plotting
import pyLDAvis
import pyLDAvis.gensim
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt

# get_ipython().run_line_magic('matplotlib', 'inline')

# Enable logging for gensim - optional 
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                   level=logging.ERROR)
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

# Prepare the stopwords
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
# stop_words.     , 'subject', 're', 'edu', 'use']) # if we wanted to extend

def sent_to_words(sentences):
    for sentence in sentences:
        yield(simple_preprocess(str(sentence), deacc=True)) # we are going to remove punctuation

data_words = list(sent_to_words(doc_pages_str))
# print (data_words)

# Build bigram and trigram models
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)
trigram= gensim.models.Phrases(bigram[data_words], threshold=100)

bigram_mod = gensim.models.phrases.Phraser (bigram)
trigram_mod = gensim.models.phrases.Phraser (trigram)


# See trigram example
print(trigram_mod[bigram_mod[data_words[0]] ])


##
## Remove Stopwords, Make Bigrams and Lemmatize
##
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts ]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram[bigram_mod[doc]] for doc in texts]

def lemmatization(texts,
                  allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])

    return texts_out

# Remove Stop Words
data_words_nostops = remove_stopwords(data_words)

# Form Bigrams
data_words_bigrams = make_bigrams(data_words_nostops)

# Do lemmatization kepping only noun, adj, vb, adv
data_lemmatized = lemmatization(data_words_bigrams)

print(data_lemmatized[:1])

##
## Create the Dictionary and Corpus needed for Topic Modeling
##
id2word = corpora.Dictionary(data_lemmatized)

# Create Corpus
texts = data_lemmatized

# Term Doc Freq
corpus = [id2word.doc2bow(text) for text in texts]
# View
print(corpus[:1]) # Gensim creates a unique id for each word in the document.

##
## Building Topic Model
##

# Latent Dirichlet (LDA) Model
lda_model = gensim.models.ldamodel.LdaModel (corpus=corpus,
                                             id2word=id2word, num_topics=20, random_state=100,
                                             update_every=1, chunksize=100, passes=10,
                                             alpha='auto', per_word_topics=True)



##
## View the topics in LDA model
##
pprint(lda_model.print_topics())
loc_lda = lda_model[corpus]


##
## Compute Model Perplexity and Coherence Score
##

# Compute perplexity a measure of how good the model is
print("\nPerplexity: ", lda_model.log_perplexity(corpus)) # A

# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word,
                                     coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print("\nCoherence Score: ", coherence_lda)

# Visualize the topics
# pyLDAvis.enable_notebook()
vis =  pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
# pyLDAvis.save_html(vis, './pyLDAvis.html')


##
## Building LDA Mallet Mode
##

malle_path = "/Users/sal.aguinaga/Boltzmann/Scriptus/notebooks/mallet-2.0.8/bin/mallet"
ldamallet = gensim.models.wrappers.LdaMallet(mallet_path=malle_path, corpus = corpus,
                                             num_topics=20, id2word=id2word)
# Show Topics
pprint(ldamallet.show_topics(formatted=True))

# Compute Coherence
coherence_model_ldmallet = CoherenceModel(model=ldamallet,
                                          texts=data_lemmatized,
                                          dictionary=id2word,
                                          coherence='c_v')
coherence_ldamallet = coherence_model_ldmallet.get_coherence()
print('\n(Mallet) Coherence Score: ', coherence_ldamallet)
