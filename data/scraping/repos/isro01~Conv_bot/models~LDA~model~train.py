import numpy as np
import matplotlib.pyplot as plt
import nltk
import re
import pandas as pd
# from pprint import pprin #for pretty print

## Gensim 

import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

## spacy for lemmatization
import spacy

import warnings 
warnings.filterwarnings("ignore" ,category= DeprecationWarning)

nltk.download('stopwords')

from nltk.corpus import stopwords
stop_words = stopwords.words('english')

extended_stopwords_list = ['from' , 'subject' ,'re' , 'edu' ,'use' ] #this depend on dataset
stop_words.extend(extended_stopwords_list)

df = pd.read_json('../dataset/newsgroups.json')

print("some examples in dataset" , df.head())

data = df.content.values.tolist()

print(data[:2])

def preprocess_data(data):
    data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]
    data = [re.sub('\s+', ' ', sent) for sent in data]
    data = [re.sub("\'", "", sent) for sent in data]

    return data

data = preprocess_data(data)
print(data[:4])

## remove puntuation and unnecessary words using gensim simple preprocess 

def gensim_preprocess(data):
  for line in data:
    yield(gensim.utils.simple_preprocess(str(line), deacc=True))  # deacc=True removes punctuations


data = list(gensim_preprocess(data))

print(data[:4])
print(type(data))

# Build the bigram and trigram models
bigram = gensim.models.Phrases(data, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data], threshold=100)  

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

print(trigram_mod[bigram_mod[data[0]]])

def remove_stopwords(data):
      for line in data:
        line = [word for word in line if word not in stop_words]
        yield(line)

def make_bigrams(data):
    return [bigram_mod[line] for line in data]

def make_trigrams(data):
    return [trigram_mod[bigram_mod[line]] for line in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

data = list(remove_stopwords(data))
data = make_bigrams(data)

# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
# !python3 -m spacy download en
nlp = spacy.load('en', disable=['parser', 'ner'])

data_lemmatized = lemmatization(data, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

print(data_lemmatized[:2])

index_to_word = corpora.Dictionary(data_lemmatized) #using gensim api to make dictionary

corpus = [index_to_word.doc2bow(line) for line in data_lemmatized]


print("Now training your lda model")

# Build LDA model
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=index_to_word,
                                           num_topics=20,
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)

model_dir = '../lda_checkpoint'

if not os.path.isdir(model_dir):
  os.mkdir(model_dir)


path = os.path.join(model_dir, 'topic_model.lda')
lda_model.save(path)
print("LDA MODEL SAVED SUCCESSFULLY")