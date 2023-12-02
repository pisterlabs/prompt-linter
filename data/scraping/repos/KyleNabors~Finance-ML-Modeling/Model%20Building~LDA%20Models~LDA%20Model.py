import os
import nltk
import sys
#Open Download Editor
#nltk.download()

import ssl
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
import pyLDAvis
import pyLDAvis.gensim
import warnings
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import nbformat
warnings.filterwarnings("ignore",category=DeprecationWarning)

#Find and import config file
config_path = os.getcwd()
sys.path.append(config_path)
import config

#Variables, Paramaters, and Pathnames needed for this script
database_file = config.database
database_folder = config.database_folder
bert_models = config.bert_models
bert_models_local = config.bert_models_local
keywords = config.keywords
lda_models = config.lda_models

Body = config.Body
Model = config.Model
Model_Subfolder = f'/{Body} Texts/{Model}'
texts_folder = config.texts
Model_Folder = texts_folder + Model_Subfolder

#Json Functions
def load_data(file):
    with open(file) as f:
        data = json.load(f)
        return data
    
def write_data(file, data):
    with open(file, 'w', encoding="utf-8") as f:
        json.dump(data, f, indent=4)
        
#Load Data
stopwords = stopwords.words('english')
#texts = load_data(f"{Model_Folder}/{Model}_texts.json")

docs = pd.read_csv(f"{Model_Folder}/{Model}_texts_long.csv") 
texts = docs['segment'].tolist()

keywords = config.keywords

#Remove Stopwords
def lemmatization(data, allowed_postags=["NOUN", "ADJ", "VERB", "ADV"]):
    nlp = spacy.load("en_core_web_lg", disable=["parser", "ner"])
    texts_out = []
    for text in texts:  # unpack the sublist into text and label
        doc = nlp(text)
        new_text = []
        for token in doc:
            if token.pos_ in allowed_postags:
                new_text.append(token.lemma_)
        final = " ".join(new_text)
        texts_out.append(final)
    return (texts_out)

#Lemmatize
lemmatized_texts = lemmatization(texts)

#Remove Stopwords
def gen_words(texts):
    final = []
    for text in texts:
        new = gensim.utils.simple_preprocess(text, deacc=True)
        final.append(new)
    return(final)

#Create Dictionary
data_words = gen_words(lemmatized_texts)

#Bigrams and Trigrams
bigrams_phrases = gensim.models.Phrases(data_words, min_count=5, threshold=10)
trigrams_phrases = gensim.models.Phrases(bigrams_phrases[data_words], threshold=10)

bigrams = gensim.models.phrases.Phraser(bigrams_phrases)
trigram = gensim.models.phrases.Phraser(trigrams_phrases)

def make_bigrams(texts):
    return([bigrams[doc] for doc in data_words])

def make_trigram(texts):
    return([trigram[bigrams[doc]] for doc in data_words])

data_bigrams = make_bigrams(data_words)
data_bigrams_trigrams = make_trigram(data_bigrams)

#TF-IDF REMOVAL
from gensim.models import TfidfModel

id2word = corpora.Dictionary(data_bigrams_trigrams)
texts = data_bigrams_trigrams
corpus = [id2word.doc2bow(text) for text in data_words]
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
    
 #Build LDA Model
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus[:-1],
                                           id2word=id2word,
                                           num_topics=8,
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=2,
                                           iterations=50,
                                           alpha="auto")    

lda_model.save(f'{lda_models}/{Body}/{Model}/LDA Model.model')

new_model = gensim.models.ldamodel.LdaModel.load(f'{lda_models}/{Body}/{Model}/LDA Model.model')
lda_model = gensim.models.ldamodel.LdaModel.load(f'{lda_models}/{Body}/{Model}/LDA Model.model')  
         
vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word, mds='mmds', R=10)
pyLDAvis.save_html(vis, f'{lda_models}/Visuals/LDA Test.html')

from collections import Counter
topics = lda_model.show_topics(formatted=False)
data_flat = [w for w_list in data_words for w in w_list]
counter = Counter(data_flat)

out = []
for i, topic in topics:
    for word, weight in topic:
        out.append([word, i , weight, counter[word]])

df = pd.DataFrame(out, columns=['word', 'topic_id', 'importance', 'word_count'])        

# Plot Word Count and Weights of Topic Keywords
fig, axes = plt.subplots(2, 4, figsize=(16,10), sharey=True, dpi=160)
cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
for i, ax in enumerate(axes.flatten()):
    ax.bar(x='word', height="word_count", data=df.loc[df.topic_id==i, :], color=cols[i], width=0.5, alpha=0.3, label='Word Count')
    ax_twin = ax.twinx()
    ax_twin.bar(x='word', height="importance", data=df.loc[df.topic_id==i, :], color=cols[i], width=0.2, label='Weights')
    ax.set_ylabel('Word Count', color=cols[i])
    ax_twin.set_ylim(0, 0.40); ax.set_ylim(2000, 60000)
    ax.set_title('Topic: ' + str(i), color=cols[i], fontsize=16)
    ax.tick_params(axis='y', left=False)
    ax.set_xticklabels(df.loc[df.topic_id==i, 'word'], rotation=30, horizontalalignment= 'right')
    ax.legend(loc='upper left'); ax_twin.legend(loc='upper right')

fig.tight_layout(w_pad=2)    
fig.suptitle('Word Count and Importance of Topic Keywords', fontsize=22, y=1.05)    
plt.show()
#plt.savefig("/Users/kylenabors/Documents/GitHub/Finance-ML-Modeling/Models/One Model/LDA Model.png")

