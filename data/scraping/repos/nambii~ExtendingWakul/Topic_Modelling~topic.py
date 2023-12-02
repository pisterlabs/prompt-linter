"""topic.py
To find the opics in the body and title using ldamulticore
"""

import os
import re
import operator
import matplotlib.pyplot as plt
import warnings
import gensim
import numpy as np
import array
import io
warnings.filterwarnings('ignore')  # Let's not pay heed to them right now
from gensim.models import CoherenceModel, LdaModel, LsiModel, HdpModel,ldamulticore
from SentenceTokeniser import SentenceTokeniser
from gensim.models.wrappers import LdaMallet
from gensim.corpora import Dictionary
from pprint import pprint
import gensim
from glove import Glove
from gensim import corpora

def topicModelling(_text,part) :
    """Function to find the topics from title and body"""
    doc =[]
    text=[]
    #preporocess the text
    text.append(_text)
    doc_clean=[SentenceTokeniser.review_to_wordlist(doc, True,part).split() for doc in text]
    doc1=" ".join(doc_clean[0])
    print("saf   ",doc_clean)
    #create a dictionary from corpora
    dictionary = corpora.Dictionary(doc_clean)
    #create a doc to term matrix
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
    #intialize the type of lda to use
    Lda = gensim.models.ldamulticore.LdaMulticore
    #for title
    if(part == 't') :
        topics_new = []
        iterations=0
        while(len(topics_new) ==0 and iterations<20) :
            #train lda a model
            ldamodel = Lda(doc_term_matrix, num_topics=6, id2word = dictionary, passes=50)#unsupervised training
            pp=ldamodel.show_topics(num_topics=6, num_words=2)
            topics=[]
            for p in pp:

                topic=" ".join(re.findall('"([^"]*)"', p[1]))
                if(topic not in topics ) :
                   topics.append(topic)
            topic=topics[0].split()
            for t in topics :
                #if the topic is concurrent words in the title add them to topics
                if(t in _text.lower()) :
                    topics_new.append(t)
            iterations=iterations+1
        return topics_new
        #for body
    else:
        topics_new = []
        topics_1=[]

        while(len(topics_new) ==0 ) :
            topics_1=[]
            ldamodel = Lda(doc_term_matrix, num_topics=4, id2word = dictionary, passes=50)
            pp=ldamodel.show_topics(num_topics=4, num_words=1)
            print(pp)
            topics=[]
            for p in pp:
                print(p[1])
                topic=" ".join(re.findall('"([^"]*)"', p[1]))
                if(topic not in topics_1 ) :
                   topics_1.append(topic)
            pp1=ldamodel.show_topics(num_topics=4, num_words=2)
            print(pp1)
            # topics=[]
            for p in pp1:
                print(p[1])
                topic=" ".join(re.findall('"([^"]*)"', p[1]))
                if(topic not in topics ) :
                   topics.append(topic)
            for t in topics :
                if(t in doc1.lower()) :
                    topics_new.append(t)
            topics=topics_1+topics_new
        return topics
def findTopics(_text,_title) :
    """Function to preprocess the title and body before finding the topic
        and return topics
    """
    topics=[]
    t=_title.split("|")
    title=t[0].split("-")
    body=_text
    body=title[0]+" "+title[0]+ " "+title[0]+ " "+ body
    topics.append(topicModelling(title[0],'t'))
    topics.append(topicModelling(body,'b'))
    topics = np.array(topics)
    topics = topics.flatten()
    print(topics)
    return(topics)
