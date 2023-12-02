
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
with open(r"E:\Helen\FinalProject_INFO5731\CSV_formatted\DS_dividedbyYears\DS1\after2020_DS1.csv", "r",  newline="", encoding='utf-8') as file:
    df1= pd.read_csv(file)

file.close()

df1['abstract']=df1['abstract'].apply(lambda x: " ".join(x for x in str(x).split() if not x.isdigit() and not x.isspace()))
df1['abstract']=df1['abstract'].str.replace('[^\w\s,]','')
#df1['abstract']=df1['abstract'].str.lower()

with open(r"E:\Helen\FinalProject_INFO5731\CSV_formatted\DS_dividedbyYears\DSApr3\after2020_newOnly_Ap3.csv", "r",  newline="", encoding='utf-8') as file:
    df2= pd.read_csv(file)

file.close()

df2['abstract']=df2['abstract'].apply(lambda x: " ".join(x for x in str(x).split() if not x.isdigit() and not x.isspace()))
df2['abstract']=df2['abstract'].str.replace('[^\w\s,]','')
#df2['abstract']=df2['abstract'].str.lower()

with open(r"E:\Helen\FinalProject_INFO5731\CSV_formatted\DS_dividedbyYears\DSApr10\after2020_newOnly_Apr10.csv", "r",  newline="", encoding='utf-8') as file:
    df3= pd.read_csv(file)

file.close()

df3['abstract']=df3['abstract'].apply(lambda x: " ".join(x for x in str(x).split() if not x.isdigit() and not x.isspace()))
df3['abstract']=df3['abstract'].str.replace('[^\w\s,]','')
#df3['abstract']=df3['abstract'].str.lower()

with open(r"E:\Helen\FinalProject_INFO5731\CSV_formatted\DS_dividedbyYears\DSApr17\after2020_newOnly_Apr17.csv", "r",  newline="", encoding='utf-8') as file:
    df4= pd.read_csv(file)

file.close()

df4['abstract']=df4['abstract'].apply(lambda x: " ".join(x for x in str(x).split() if not x.isdigit() and not x.isspace()))
df4['abstract']=df4['abstract'].str.replace('[^\w\s,]','')
#df4['abstract']=df4['abstract'].str.lower()


# Topic modeling with LDA and Gensim
tokenizer = RegexpTokenizer(r'\w+')

# create English stop words list
en_stop = get_stop_words('en')
stop_plus = ['word', 'count', 'text', 'all', 'right', 'no', 'without', 'abstract', 'no', 'reuse', 'without', 'abstract', 'nan']

# Create PorterStemmer
p_stemmer = PorterStemmer()
# create list of documents
abstract_set = []
for abstract in df1['abstract'][:2].dropna():
    abstract_set.append(abstract)

for abstract in df2['abstract'][:2].dropna():
    abstract_set.append(abstract)

for abstract in df3['abstract'][:2].dropna():
    abstract_set.append(abstract)

for abstract in df4['abstract'][:2].dropna():
    abstract_set.append(abstract)

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
    stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens if len(i)>3]
    texts.append(stemmed_tokens)

# turn our tokenized documents into a id <-> term dictionary
dictionary = corpora.Dictionary(texts)
#print(dictionary)
# convert tokenized documents into a document-term matrix
corpus = [dictionary.doc2bow(text) for text in texts]
#print(corpus)

# setting up our imports

from gensim.models import ldaseqmodel
from gensim.corpora import Dictionary, bleicorpus
import numpy
from gensim.matutils import hellinger

time_slice = [len(df1), len(df2), len(df3), len(df4)]
ldaseq = ldaseqmodel.LdaSeqModel(corpus=corpus, id2word=dictionary, time_slice=time_slice, num_topics=30)


topics = []
i=0
while i <4:
    topic_i=[]
    for topic in ldaseq.print_topics(time=i, top_terms=12):
        topic_i.append(topic)
    topics.append(topic_i)
    i+=1


df_topics = pd.DataFrame(topics)

with open (r"E:\Helen\FinalProject_INFO5731\ALL_OUTPUTS\PTM_timeslice.csv", 'w',  newline="",
         encoding='utf-8') as file:
    df_topics.to_csv(file)


from gensim.models.coherencemodel import CoherenceModel

# we just have to specify the time-slice we want to find coherence for.
topics_dtm = ldaseq.dtm_coherence(time=2)
cm_DTM = CoherenceModel(topics=topics_dtm, texts=texts, dictionary=dictionary, coherence='c_v')

print ("DTM Python coherence is", cm_DTM.get_coherence())
