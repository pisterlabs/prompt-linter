# Apply GENSIM

import pandas as pd
import csv
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import spacy
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt

# open metadata file and cleaning
with open(r"E:\Helen\FinalProject_INFO5731\All_DS_CORD19\DS_CORD19_1st\all_sources_metadata_2020-03-13.csv", "r",  newline="", encoding='utf-8') as file:
    df= pd.read_csv(file)
file.close()

df['abstract']=df['abstract'].apply(lambda x: " ".join(x for x in str(x).split() if not x.isdigit() and not x.isspace()))
df['abstract']=df['abstract'].str.replace('[^\w\s,]','')
df['abstract']=df['abstract'].str.lower()

# Topic modeling with LDA and Gensim
tokenizer = RegexpTokenizer(r'\w+')

# create English stop words list
en_stop = get_stop_words('en')
stop_plus = ['word', 'count', 'text', 'all', 'right', 'no', 'without', 'abstract', 'no', 'reuse', 'without']

# Create PorterStemmer
p_stemmer = PorterStemmer()

# create list of documents
abstract_set = []
for abstract in df['abstract'].dropna():
    abstract_set.append(abstract)
# print(abstract_set)

# list for tokenized documents in loop
texts = []

# loop through document list
for i in abstract_set:
    # clean and tokenize document string
    raw = i.lower()
    tokens = tokenizer.tokenize(raw)
    # remove stop words from tokens
    stopped_tokens = [i for i in tokens if not i in en_stop + stop_plus]
    # stem tokens
    stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
    texts.append(stemmed_tokens)

# turn our tokenized documents into a id <-> term dictionary
dictionary = corpora.Dictionary(texts)
#print(dictionary)
# convert tokenized documents into a document-term matrix
corpus = [dictionary.doc2bow(text) for text in texts]
#print(corpus)

# generate LDA model
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=100, id2word = dictionary, passes=60)

topics = []
for topic in ldamodel.print_topics(num_topics=100,num_words=7):
    topics.append(topic)
print(topics)

df_topics = pd.DataFrame(topics)

#with open (r"E:\Helen\FinalProject_INFO5731\ALL_OUTPUTS\DS_1st\Content_analysis\LDAgensim_TopicModeling_DS1.csv", 'w',  newline="",
#          encoding='utf-8') as file:
#    df_topics.to_csv(file)

# Compute Model Perplexity and Coherence Score: This model to judge how good the model performed,
# especially by Coherene score

# Compute Perplexity
print('Perplexity: ', ldamodel.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

# Compute Coherence Score
coherence_ldamodel = CoherenceModel(model=ldamodel, texts=texts, dictionary=dictionary, coherence='c_v')
coherence_lda = coherence_ldamodel.get_coherence()
print('\nCoherence Score: ', coherence_lda)