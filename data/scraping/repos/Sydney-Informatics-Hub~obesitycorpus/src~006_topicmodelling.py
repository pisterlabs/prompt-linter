# Critical note:
# Topic modelling was optimised on the NON-DEDUPLICATED corpus; however, this resulted in more
# "reasonable" topics than implemented on the deduplicated one, so the model used below comes from 
# the full cleaned but not deduplicated corpus.
# it is being APPLIED in this document to the deduplicated corpus.
# The optimal model on the deduplicated corpus had 18 topics and is also provided.
# %%
import pathlib
from utils import get_projectpaths
(projectroot, rawdatapath, cleandatapath, processeddatapath) = get_projectpaths()
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
import matplotlib.pyplot as plt
# %matplotlib inline
# Enable logging for gensim - optional
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
import pickle
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
# spacy for lemmatization
import spacy
from pprint import pprint

#%%
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
corpusdf = pd.read_pickle(processeddatapath/'corpusdf_deduped_by_source.pickle')
print(corpusdf.shape)


# %% 
# Convert body to list
bodies = corpusdf.body.values.tolist()
# Remove new lines
bodies = [re.sub('\s+', ' ', sent) for sent in bodies]
# Remove single quotes
bodies = [re.sub("\'", "", sent) for sent in bodies]
# Remove double quotes
bodies = [re.sub('"', "", sent) for sent in bodies]
bodies_words = list(sent_to_words(bodies))
# Remove Stop Words
bodies_words_nostops = remove_stopwords(bodies_words)

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
lda_model =  gensim.models.LdaModel.load(str(processeddatapath/'topicmodels/full_m_17_model'))


# %% Print the Keyword in the topics
pprint(lda_model.print_topics())

# %% Compute Perplexity
# Perplexity:  -8.685147984244903
print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

#%%
def format_topics_texts(ldamodel=lda_model, corpus=corpus, texts=bodies):
    # Init output
    topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        topicproblist = sorted(row[0], key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(topicproblist):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                # note: adding +1 (ONE) here to make topics match what is seen in the ldavis visualisation above
                topics_df = topics_df.append(pd.Series([topic_num+1, round(prop_topic,4)]), ignore_index=True)
            else:
                break
    topics_df.columns = ['dominant_topic', 'percent_contribution']

    return(topics_df)

df_topics_keywords = format_topics_texts(ldamodel=lda_model, corpus=corpus, texts=bodies)

# Format
df_dominant_topic = df_topics_keywords.reset_index()
df_dominant_topic.columns = ['article_number', 'dominant_topic', 'percent_contribution']

# Show
df_dominant_topic = df_dominant_topic.drop('article_number', axis=1)
# df_dominant_topic.head(10)

# cbind the corpus and topic df
corpusdf_with_topics = pd.concat([corpusdf.reset_index(drop=True), df_dominant_topic], axis=1)
# %% export 
pd.to_pickle(corpusdf_with_topics, processeddatapath/'corpusdf_with_topics.pickle')
print(corpusdf_with_topics.shape)
