from nltk.tokenize import word_tokenize
from flask import Flask, redirect, url_for, request
from nltk.corpus import wordnet
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import os.path
from gensim import corpora
from nltk.corpus import stopwords
from gensim.models import LsiModel
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt
import pandas

nltk.download('book')

def load_data(path, file_name):
    documents_list = []
    titles = []
    with open(os.path.join(path, file_name), "r") as fin:
        for line in fin.readlines():
            text = line.strip()
            documents_list.append(text)
    titles.append(text[0:min(len(text), 100)])
    return documents_list, titles


def preprocess_data(doc_set):
    tokenizer = RegexpTokenizer(r'\w+')
    en_stop = set(stopwords.words('english'))
    p_stemmer = PorterStemmer()
    texts = []
    for i in doc_set:
        raw = i.lower()
        tokens = tokenizer.tokenize(raw)
        stopped_tokens = [i for i in tokens if not i in en_stop]
        stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
        texts.append(stemmed_tokens)
    return texts


def prepare_corpus(doc_clean):
    dictionary = corpora.Dictionary(doc_clean)
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
    return dictionary, doc_term_matrix


def create_gensim_lsa_model(doc_clean, number_of_topics, words):
    dictionary, doc_term_matrix = prepare_corpus(doc_clean)
    lsamodel = LsiModel(
        doc_term_matrix, num_topics=number_of_topics, id2word=dictionary)
    return lsamodel


def compute_coherence_values(dictionary, doc_term_matrix, doc_clean, stop, start=2, step=3):
    coherence_values = []
    model_list = []
    for num_topics in range(start, stop, step):
        model = LsiModel(doc_term_matrix, num_topics=number_of_topics,
                         id2word=dictionary)  # train model
        model_list.append(model)
        coherencemodel = CoherenceModel(
            model=model, texts=doc_clean, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
    return model_list, coherence_values


def stackfind(idea):
  nltk.download("book")
  nltk.download("omw")
  ml=[]

  idea=word_tokenize(idea)
  stop_words=set(stopwords.words("english"))
  filtered=[]
  for w in idea:
    if w not in stop_words:
      filtered.append(w)

  lemmatizer=WordNetLemmatizer()
  stemmed=[]
  for w in filtered:
    stemmed.append(lemmatizer.lemmatize(w))

  L1=[] #opencv
  synonymns1=[]

  for i in wordnet.synsets("image"):
    for l in i.lemmas():
      synonymns1.append(l.name())
  for i in wordnet.synsets("gesture"):
    for l in i.lemmas():
      synonymns1.append(l.name())


  i1=set(stemmed).intersection(synonymns1)
  if len(i1)>0:
    ml.append("opencv")

  L2=["video","detect","text"] #teseract
  synonymns2=[]
  for x in L2:
    for i in wordnet.synsets(x):
      for l in i.lemmas():
        synonymns2.append(l.name())

  i2=set(stemmed).intersection(L2)
  if len(i2)==3:
    ml.append("teseract")

  L4=["app","mobile"] #AppDev
  synonymns4=[]
  for x in L4:
    for i in wordnet.synsets(x):
      for l in i.lemmas():
        synonymns4.append(l.name())

  i4=set(stemmed).intersection(synonymns4)

  if len(i4)>0:
    ml.append("AppDev","Flutter","Android Studio")

  L3=["video","edit","editing"] #AE
  synonymns3=[]
  for x in L3:
    for i in wordnet.synsets(x):
      for l in i.lemmas():
        synonymns3.append(l.name())

  i3=set(stemmed).intersection(synonymns3)
  if len(i3)>=2:
    ml.append("AfterEffects")

  L5=["web","site","website","scrape"] #Website
  synonymns5=[]
  for x in L5:
    for i in wordnet.synsets(x):
      for l in i.lemmas():
        synonymns5.append(l.name())
    

  i5=set(stemmed).intersection(synonymns5)
  if len(i5)>=1:
    ml.append("HTML")
    ml.append("CSS")
    ml.append("JS")

  L8=["website","background"] #qaunta.js
  synonymns8=[]
  for x in L8:
    for i in wordnet.synsets(x): 
      for l in i.lemmas():
        synonymns8.append(l.name())
  
  i8=set(stemmed).intersection(synonymns8)
  if len(i8)>=2:
    ml.append("quanta.js") 

  L10=["fullstack","functional","responsive","web","website"] #Website
  synonymns10=[]
  for x in L10:
    for i in wordnet.synsets(x):
      for l in i.lemmas():
        synonymns10.append(l.name())

  i10=set(stemmed).intersection(synonymns10)

  if len(i10)>=1:
    ml.append("Django")
    ml.append("Node.js")
    ml.append("Bootstrap")
  
  L11=["text","recommendation"] #nltk
  synonymns11=[]  
  for x in L11:
    for i in wordnet.synsets(x):
      for l in i.lemmas():
        synonymns11.append(l.name())
  i11=set(stemmed).intersection(synonymns11)
  if len(i11)>=1:
    ml.append("nltk")

  return ml
