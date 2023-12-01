# This was run on Artemis HPC
# %%
import re
import numpy as np
import pandas as pd
# silence annoying warning
pd.options.mode.chained_assignment = None  # default='warn'
# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
# spacy for lemmatization
import spacy
import matplotlib.pyplot as plt
#%matplotlib inline
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
import pickle
import os
#
import nltk
from nltk.corpus import stopwords
stop_words = stopwords.words('english')


# %%
import sys
num_topics = sys.argv[1]
print (sys.version)

# %% FUNCTION DEFINITION BLOCK ------------------------------------------------
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def lemmatize(texts, allowed_postags=['PROPN','NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations




# WORK BLOCK ------------------------------------------------
# %% load data and obesity names
corpusdf = pd.read_csv("corpusdf_deduped_by_source.csv")

# %%
def replace_patterns(patterns, replacements, bodies):
    for idx, val in enumerate(patterns):
        bodies = [re.sub(val, replacements[idx], sent) for sent in bodies]
    return(bodies)
    
# Convert body to list
bodies = corpusdf.body.values.tolist()
# Remove new lines, single and double quotes
bodies = replace_patterns(
    patterns = ['\s+','\'','"'],
    replacements = [' ', '', ''],
    bodies = bodies
)
bodies_words = list(sent_to_words(bodies))
# Remove Stop Words
bodies_words_nostops = remove_stopwords(bodies_words)

# %%
# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
# python3 -m spacy download en
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

# Do lemmatization keeping only noun/propn, adj, vb, adv
bodies_lemmatized = lemmatize(bodies_words_nostops, allowed_postags=['PROPN','NOUN', 'ADJ', 'VERB', 'ADV'])



# %% Create Dictionary
corpusdict = corpora.Dictionary(bodies_lemmatized)
# print how many words are in the dictionary
# gensim mem usage will be 24 * num_topics * this
print(corpusdict)


# Term Document Frequency
corpus = [corpusdict.doc2bow(text) for text in bodies_lemmatized]

# %%
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                          id2word=corpusdict,
                          num_topics=num_topics,
			              alpha='auto',
                          random_state=42,
                          per_word_topics=True)

temp_file = f'm_{num_topics}_model'
lda_model.save(temp_file)

coherence_model_lda = CoherenceModel(model=lda_model, texts=bodies_lemmatized, dictionary=corpusdict, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
outcoh = f"m_{num_topics}_coherence"
with open(outcoh, 'w') as f:
    f.write(str(coherence_lda))