# bigrams and trigrams not working in gensim
# https://www.youtube.com/watch?v=TKjjlp5_r7o&list=PL2VXyKi-KpYttggRATQVmgFcQst3z6OlX&index=10

# import libraries
import numpy as np
import json
import glob
import regex as re
import string
from os import path

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models.coherencemodel import CoherenceModel

# Spacy
import spacy
import nltk
# nltk.download("stopwords")
# nltk.download("punkt")

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# VIS
# !pip install pyLDAvis
import pyLDAvis
import pyLDAvis.gensim_models

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# load json data
def append_data(filename):

    listObj = []

    # Check if file exists
    if path.isfile(filename) is False:
      raise Exception("File not found")
 
    # Read JSON file
    with open(filename) as fp:
        listObj = json.load(fp)
 
    print('Successfully loaded JSON file\n')        

    return listObj

# load specific key from file
savedJobs = append_data("C:\\Users\\bminn\\Documents\\PROJECT INVESTIGATION\\NLP\\JOB POST\\SavedJobPosts.json")

# remove http unicode characters
def removeUnicodeChars(text):

    charDict = {"\u2013": "-",
                "\u2026": "...",
                "\u2019": "'",
                "\u0142": "l",
                "\u0105": "a",
                "\u0119": "e",
                "\u2014": "-",
                "\u2695": " ",
                "\u0101": "a",
                "\u0113": "e",
                "\u0144": "n",
                "\u2018": "'",
                "\u2022": " "}

    for key in charDict:    
        text = re.sub(key, charDict[key], text)
    

    return text

# clean the job posts main data
def clean_docs(_savedJobs):

    _cleaned_posts = []
    _position_posts = []

    for dict in _savedJobs:        
        text_with_bad_chars = dict['post']

        text = removeUnicodeChars(text_with_bad_chars)

        # remove punctuation
        unpunctuated = text.translate(str.maketrans("", "", string.punctuation))
        # get rid of all numbers
        numbers_removed = "".join([i for i  in unpunctuated if not i.isdigit()])


        stop_words = set(stopwords.words("english"))

        # split text into single words to remove stop words
        word_tokens =  word_tokenize(numbers_removed)

        filtered_sentences = [w for w in word_tokens if not w.lower() in stop_words] # returns array of text

        _cleaned_posts.append(" ".join(filtered_sentences))
        _position_posts.append(" ".join(dict['position']))

    return _cleaned_posts, _position_posts


#reducing words to their core root word. also reducing complexity of sentences down to allowed postags only.
def lemmatization(texts, allowed_postags=['NOUN ', 'ADJ', 'VERB', 'ADV']):
  nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])

  texts_out = []

  for text in texts:
    doc = nlp(text)
    new_text = []
    for token in doc:
      if token.pos_ in allowed_postags:
        new_text.append(token.lemma_)

    final = " ".join(new_text)
    texts_out.append(final)

    return texts_out


def gen_words(texts):
  final = []
  for text in texts:
    new = gensim.utils.simple_preprocess(text, deacc=True)
    final.append(new)

  return final

# Main Code
cleaned_posts, position_posts = clean_docs(savedJobs)

lemmatized_texts = lemmatization(cleaned_posts)

print("orignal data:\n")
print(cleaned_posts[0][0:90])
print("\nlemmatized data:\n")
print(lemmatized_texts[0][0:90])

data_words = gen_words(lemmatized_texts)
print("\ngensim data:\n")
print(data_words[0][0:90])

# Build the bigram and trigram models
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

data_bigrams = make_bigrams(data_words)
data_trigrams = make_trigrams(data_bigrams)


# See trigram example
print("\nbigrams data:\n")
print(data_trigrams)


# create a dict with words and their frequency

id2word = corpora.Dictionary(data_words)

corpus = []
for text in data_words:
  new = id2word.doc2bow(text)
  corpus.append(new)


print("\ncorpus data:\n")
print(corpus[0][0:90])

# chekc the first word
word = id2word[[0][:1][0]]
print(word)

lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                            id2word=id2word,
                                            num_topics=100,
                                            update_every=1,
                                            chunksize=100,
                                            passes=10,
                                            alpha="auto")

pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word, mds="mmds", R=30)
vis