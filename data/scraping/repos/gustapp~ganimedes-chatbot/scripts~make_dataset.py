#%%
import pandas as pd
import click

df = pd.read_json('./db_poli_v2.json')
df = df[['sigla', 'name', 'objetivos', 'programa', 'programa_resumido']]
df['name'] = df['name'].apply(lambda x: '-'.join(x.split('-')[1:]).strip())

documents = df[['name', 'objetivos', 'programa', 'programa_resumido']].apply(lambda x: ' '.join(x), axis=1)

print(len(documents))
df.head()

#%% [markdown]
# ## Pre-processing
import nltk
import gensim
from gensim.corpora import Dictionary
from gensim import corpora, models

from nltk import collocations
from nltk.tokenize import RegexpTokenizer
from nltk.stem import RSLPStemmer
from nltk.corpus import stopwords

nltk.download('rslp')

STOPWORDS = stopwords.words('portuguese')

rgtokenizer = RegexpTokenizer(r'\w+')

#%% [markdown]
# ### Lemmatize & Stemming

#%%
# Preprocessing methods
def lemmatize_stemming(text):
    stemmer = RSLPStemmer()
    return stemmer.stem(text)

def preprocess(text, word_tokenize=rgtokenizer.tokenize):
    result = []
    for token in word_tokenize(text):
        if token not in STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result

#%%
# (Debug) Preview a doc before prepocessing
doc_sample = documents[0]

print('original document: ')
words = []
for word in doc_sample.split(' '):
    words.append(word)
print(words)
print('\n\n tokenized and lemmatized document: ')
print(preprocess(doc_sample))

#%%
# Preprocess all documents
cleaned_docs = documents.map(preprocess)
cleaned_docs[:10]

#%% [markdown]
# ### Collocations

#%%
# Build Abstract Text Corpus
corpus_abstract = ' '.join(cleaned_docs.map(lambda tokens: ' '.join(tokens)))
len(corpus_abstract.split(' '))

#%%
# Identify Collocations
cl = collocations.BigramCollocationFinder.from_words(corpus_abstract.split(' '))

#%%
# 85 Best Collocations by likelihood ratio
set_collocation = cl.nbest(collocations.BigramAssocMeasures().likelihood_ratio, 200)
set_collocation[:10]

#%%
# Apply Collocations
def apply_collocations(tokens, set_colloc=set_collocation):
    """ Reference: acidtobi
        url: https://stackoverflow.com/questions/43572898/apply-collocation-from-listo-of-bigrams-with-nltk-in-python
    """
    res = ' '.join(tokens)
    for b1,b2 in set_colloc:
        res = res.replace("%s %s" % (b1 ,b2), "%s_%s" % (b1 ,b2))
    for b1, b2 in set_colloc:
        res = res.replace("_%s_%s" % (b1 ,b2), "_%s %s_%s" % (b1, b1 ,b2))
    return res.split(' ')

processed_docs = cleaned_docs.map(apply_collocations)
processed_docs[:10]

#%% [markdown]
# ### Dictionary

#%%
# Create a dictionary from ‘processed_docs’ containing the number of times a word appears in the training set
dictionary = Dictionary(processed_docs)

count = 0
for k, v in dictionary.iteritems():
    print(k, v)
    count += 1
    if count > 10:
        break

#%%
# Filter out tokens that appear in less than 2 docs or more than 0.7 docs
# Keep (at most) only the first 100000 most frequent
dictionary.filter_extremes(no_below=2, no_above=0.7, keep_n=100000)
len(dictionary.token2id)

#%% [markdown]
# ### Encoding Model (BoW, TF-IDF)

#%%
# Gensim doc2bow
# For each document we create a dictionary reporting how many
# words and how many times those words appear
bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
bow_corpus[0]

#%%
# (Debug) Preview BOW for our sample 
bow_doc_sample_number = bow_corpus[0]

for i in range(len(bow_doc_sample_number)):
    print("Word {} (\"{}\") appears {} time.".format(bow_doc_sample_number[i][0], 
            dictionary[bow_doc_sample_number[i][0]], bow_doc_sample_number[i][1]))

#%%
# TF-IDF
# Preview TF-IDF scores for our first doc
tfidf = models.TfidfModel(bow_corpus)
corpus_tfidf = tfidf[bow_corpus]

from pprint import pprint

for doc in corpus_tfidf:
    pprint(doc)
    break

# #%% [markdown]
# # ## Topic Model

# #%%
# # Running LDA using BOW
# from gensim.models.coherencemodel import CoherenceModel

# import logging
# logging.basicConfig(filename='./lda_bow.log',
#                     format="%(asctime)s:%(levelname)s:%(message)s",
#                     level=logging.INFO)

# coherence_x_n_topics = []
# for n_topics in range(3, 101):
#     lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=n_topics, id2word=dictionary, \
#         alpha=[0.01]*n_topics, eta=[0.01]*len(dictionary.keys()), passes=78, workers=11)

#     lda_model.save('./lda_bow/lda_model_tp-' + str(n_topics))

#     """ Coherence Model - Umass """
#     cm_bow = CoherenceModel(model=lda_model, corpus=bow_corpus, coherence='u_mass')
#     coherence_value_umass = cm_bow.get_coherence()

#     logging.info('coherence value - umass: ' + str(coherence_value_umass))

#     """ Coherence Model - C_V """
#     cm_bow_cv = CoherenceModel(model=lda_model, corpus=bow_corpus, texts=processed_docs, dictionary=dictionary, coherence='c_v')
#     coherence_value_cv = cm_bow_cv.get_coherence()

#     logging.info('coherence value - cv: ' + str(coherence_value_cv))

#     """ Coherence Model - C_UCI """
#     cm_bow_uci = CoherenceModel(model=lda_model, corpus=bow_corpus, texts=processed_docs, dictionary=dictionary, coherence='c_uci')
#     coherence_value_cuci = cm_bow_uci.get_coherence()

#     logging.info('coherence value - cuci: ' + str(coherence_value_cuci))

#     """ Coherence Model - C_NPMI """
#     cm_bow_npmi = CoherenceModel(model=lda_model, corpus=bow_corpus, texts=processed_docs, dictionary=dictionary, coherence='c_npmi')
#     coherence_value_cnpmi = cm_bow_npmi.get_coherence()

#     logging.info('coherence value - cnpmi: ' + str(coherence_value_cnpmi))

#     coherence_x_n_topics.append((n_topics, coherence_value_umass, coherence_value_cv, coherence_value_cuci, coherence_value_cnpmi))

#     for idx, topic in lda_model.print_topics(-1):
#         print('Topic: {} \nWords: {}'.format(idx, topic))

# model_metrics = pd.DataFrame(data=coherence_x_n_topics, columns=['n topics', 'umass', 'cv', 'cuci', 'cnpmi'], index=range(3, 101))
# model_metrics.head()

# model_metrics.to_csv('./coherence_curve_bow.csv') 


#%%
