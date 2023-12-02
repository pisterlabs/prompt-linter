# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 20:58:27 2019

@author: Hong.Wen.Tai
"""

import pandas as pd
import numpy as np
import re # We clean text using regex
import csv # To read the csv
from collections import defaultdict # For accumlating values
from nltk.corpus import stopwords # To remove stopwords
from gensim import corpora # To create corpus and dictionary for the LDA model
from gensim.models import LdaModel # To use the LDA model
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import tokenize
from nltk.stem import WordNetLemmatizer
import gensim
from gensim.models import CoherenceModel
import spacy
from spacy import displacy
import en_core_web_sm
from datetime import datetime
from gensim.models.phrases import Phrases, Phraser
import random
from spacy.util import minibatch, compounding
from spacy.matcher import Matcher
import seaborn as sns
import matplotlib.pyplot as plt

#Function to print clean string for visualization
def print_clean(string):
    print ('"{}"'.format(string)

#Functions for Sentiment Analysis
def sentSentiment(sentence):
    sentence_list = tokenize.sent_tokenize(sentence)
    paragraphSentiments = 0.0
    for sentence in sentence_list:
        vs = analyser.polarity_scores(sentence)
        print("{:-<69} {}".format(sentence, str(vs["compound"])))
        paragraphSentiments += vs["compound"]
    finalSentiment = round(paragraphSentiments / len(sentence_list), 4)
    print("AVERAGE SENTIMENT FOR PARAGRAPH: \t" + str(finalSentiment))
    print("----------------------------------------------------")
    return finalSentiment

#Functions for Entity Extraction
def count_profben(content_series):
    benseries = " ".join(content_series).lower()               #join Series into one string, convert to lowercase
    benlist = re.findall("prof.?leong|prof.?ben",benseries)    #include .? to identify cases such as "profben" or "prof.ben"
    return len(benlist)

def extract_mods_courses(content_series):
    regstring = " ".join(content_series).lower()         #join Series into one string, convert to lowercase (since CS1010 is the same as cs1010)
    reglist = re.findall("(([a-z|A-Z]{2,3})\d{4}[a-z|A-Z]?)",regstring)  #Use [a-z|A-Z] since \w will extract numbers too. Add brackets to extract group (module) and sub-group (course)
    modulelist, courselist = zip(*reglist)               #Unpack tuple of pairs into two tuples
    moduleseries = pd.Series(modulelist)
    courseseries = pd.Series(courselist)
    return moduleseries, courseseries

def extract_mods_courses_accurate(content_string):
    content_string = content_string.lower()    #convert string to lowercase
    reglist = re.findall("(([a-z|A-Z]{2,3})\d{4}[a-z|A-Z]?)",content_string)   #identify all modules in the string
    reglist_unique = list(set(reglist))      #remove duplicates
    
    if len(reglist_unique) == 0:            #if can't identify any modules, return NA
        return "NA"
    else: 
        module, course = zip(*reglist_unique)   #unpack tuple of pairs into two tuples: first tuple contains all mods, second contains all courses
        modulelist = []
        courselist = []
        for i in module:                   #for each mod in tuple, add to module list
            modulelist.append(i)
        for j in course:
            courselist.append(j)
        return modulelist,courselist       #return tuple:(list of mods, list of courses)
    
def extract_mod(tuple):
    modlist = tuple[0]                    #return list of mods
    modstring = " ".join(modlist)         #combine list into string of mods
    return modstring

def extract_course(tuple):
    courselist = tuple[1]
    coursestring = " ".join(courselist)
    return coursestring

#Functions for Topic Modeling
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  
        
def convert_bigram(contentseries):                #takes in a Series, fits bigram model and processing pipeline on it, returns list ready for topic modeling
    newdoc2= contentseries.tolist()
    sentence_stream_newdoc2 = [doc.split(".") for doc in newdoc2]
    
    a = []
    for doc in sentence_stream_newdoc2:
        for sentence in doc:
            a.append(sentence.split(" "))
    
    phrases_newdoc2 = Phrases(a, min_count=3, threshold=69) 
    bigram_newdoc2 = Phraser(phrases_newdoc2)
    
    data_words = list(sent_to_words(contentseries))
    data_words2 = [bigram_newdoc2[i] for i in data_words]
    doc_list =  [nlp2(" ".join(doc)) for doc in data_words2]
    return doc_list
    
def model_topic(doc_list):   #takes in list, return ldamodel and dataframe of top 10 topics
    words = corpora.Dictionary(doc_list)
    corpus = [words.doc2bow(doc) for doc in doc_list]
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=words, num_topics=5, random_state=42, 
                                            update_every=1, chunksize=100, passes=50, alpha='auto', 
                                            per_word_topics=True, dtype=np.float64)
    
    topic_dict = {};
    for i in range(5):
        topic_list = lda_model.show_topic(i, topn = 20)
        topic_dict['Topic # ' + '{:02d}'.format(i+1)] = [i[0] for i in topic_list]
    return lda_model, corpus, pd.DataFrame(topic_dict)


def assign_topic(original_dataframe, ldamodel, corpus):  #assign topic to each row in original dataframe
    original_dataframe["topic"] = 0
    original_dataframe["topic_probability"] = 0

    #pass corpus into lda_model to get probabilities of each doc belonging to each topic
    for i, row in enumerate(ldamodel[corpus]):
      # print (i)
      # print (row[0])
        b = 0
        for j in row[0]:        
            if j[1] > b:
                a = j[0] + 1       #assign topic to a (+1 since number starts from 0), probability to b. If current probability > previous probability, overwrite b
                b = j[1]     

        print(a,b)   
        original_dataframe["topic"].iloc[i]=a
        original_dataframe["topic_probability"].iloc[i]= b
    
    return original_dataframe

#Functions for spacy preprocessing pipeline:
def lemmatizer(doc, allowed_postags=['PROPN', 'NOUN', 'ADJ', 'VERB', 'ADV', 'NUM','X']):
    # This takes in a doc of tokens from the NER and lemmatizes them. 
    # Pronouns (like "I" and "you" get lemmatized to '-PRON-', so I'm removing those.
    doc = [token.lemma_ for token in doc if ((token.pos_ in allowed_postags) and (token.lemma_ != '-PRON-'))]
    doc = u' '.join(doc)
    return nlp2.make_doc(doc)

def remove_stopwords(doc):
    # This will remove stopwords and punctuation.
    # Use token.text to return strings, which we'll need for Gensim.
    doc = [token.text for token in doc if token.is_stop != True]
           #and token.is_punct != True]
    return doc

nlp2 = spacy.load(r"C:\Users\Hong.Wen.Tai\Downloads\nuswhispers_text_analysis\models")    
nlp2.add_pipe(lemmatizer,name='lemmatizer',after='ner')                #add back custom components to model
nlp2.add_pipe(remove_stopwords, name="stopwords", after = "lemmatizer")