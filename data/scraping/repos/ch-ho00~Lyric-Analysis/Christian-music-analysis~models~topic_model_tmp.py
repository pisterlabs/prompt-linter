import gensim
import gensim.corpora as corpora
from gensim.test.utils import datapath
from gensim.models import CoherenceModel, LdaModel, LsiModel, HdpModel
from gensim.models.wrappers import LdaMallet
from gensim.corpora import Dictionary
from pprint import pprint
import pyLDAvis.gensim

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

import unidecode
import codecs
import matplotlib.pyplot as plt
#%matplotlib inline
import numpy as np
import os
import re
import operator
import warnings


def generate_corpus(dictionary):
    # Create dictionary
    dic = corpora.Dictionary(dictionary)
    # Create bag of words
    corpus = [dic.doc2bow(text) for text in dictionary]
    return corpus, dic
  
def create_lda(num_topic, dictionary):
    print("__________________________Create LDA_________________________")
    corpus, dic= generate_corpus(dictionary)
    lda= gensim.models.ldamodel.LdaModel(corpus, num_topics = num_topic, id2word=dic, passes=15)
    topics = lda.print_topics(num_words = 7)
    # see list of topics
    for topic in topics:
        print(topic)
        # Save model to disk.
#    temp_file = datapath("./models")
#    lda.save(temp_file)
    return lda, dic
def text2topic(lda, dic, text):
    converted = corpora.Dictionary(text)
    other_corpus = [dic.doc2bow(word) for word in text] 
    vector = lda[other_corpus]  # get topic probability distribution for a document
    for topic in vector:
      print(topic)
    # y_axis = []
    # x_axis = []
    # for dist in vector:
        
    #     x_axis.append(dist + 1)
    #     y_axis.append(dist)
    # width = 1
    # plt.bar(x_axis, y_axis, width, align='center', color='r')
    # plt.xlabel('Topics')
    # plt.ylabel('Probability')
    # plt.title('Topic Distribution for doc')
    # plt.xticks(np.arange(2, len(x_axis), 2), rotation='vertical', fontsize=7)
    # plt.subplots_adjust(bottom=0.2)
    # plt.ylim([0, np.max(y_axis) + .01])
    # plt.xlim([0, len(x_axis) + 1])
    # plt.close()
    return vector 
def lemmatize(lyric_list,save_dir):
    # gensim Wordnet Lemmatizer
    lemmatizer = WordNetLemmatizer()
    dictionary = []
    add = []
    stop_words = set(stopwords.words('english'))
    stop_words.add('?')
    stop_words.add(',')
    stop_words.add('.')
    stop_words.add('!')
    stop_words.add('...')
    stop_words.add("\'m")
    stop_words.add("\'s")
    stop_words.add("\'re")
    stop_words.add("\'t")
    stop_words.add("'")
    stop_words.add('``')
    stop_words.add("'ll")
    
    stop_words.add("chorus")
    for lyric in lyric_list:
        lyric_list[lyric] = unidecode.unidecode(lyric_list[lyric].lower())
        add = word_tokenize(lyric_list[lyric])
        add = pos_tag(add)
        tmp = []
        for i in add:
            if not i[0] in stop_words and len(i[0]) > 3:
                if i[1] == "VB" or i[1] == "VBD" or i[1] == "VBG" or i[1]== "VBN" or i[1] == "VBZ":
                    print(i[0], lemmatizer.lemmatize(i[0], "v"))
                    tmp.append(lemmatizer.lemmatize(i[0], "v"))
                else:
                    tmp.append(i[0])
        dictionary.append(tmp)
    print(dictionary)
    copy = dictionary 
    #save lemmatized dictionary 
    filename = "dictionary.txt"
    print(str(len(dictionary))+ " lemmatized songs saved ")
    with codecs.open(save_dir+ "/" + filename, 'w', encoding='utf8') as f:
        for i in dictionary:
            f.write(str(i))
            f.write('\n')
            f.write('\n')
    return copy 
def lemmatize_unseen(lyric_list):
    # gensim Wordnet Lemmatizer
    lemmatizer = WordNetLemmatizer()
    dictionary = []
    add = []
    stop_words = set(stopwords.words('english'))
    stop_words.add('?')
    stop_words.add(',')
    stop_words.add('.')
    stop_words.add('!')
    stop_words.add('...')
    stop_words.add(':')
    stop_words.add("\'m")
    stop_words.add("\'s")
    stop_words.add("\'re")
    stop_words.add("\'t")
    stop_words.add("chorus")
    for lyric in lyric_list:
        lyric_list[lyric] = unidecode.unidecode(lyric_list[lyric].lower())
        add = word_tokenize(lyric_list[lyric])
        add = pos_tag(add)
        tmp = []
        for i in add:
            if not i[0] in stop_words and len(i[0]) >3 :
                if i[1] == "VB" or i[1] == "VBD" or i[1] == "VBG" or i[1]== "VBN" or i[1] == "VBZ":
                    tmp.append(lemmatizer.lemmatize(i[0], "v"))
                else:
                    tmp.append(i[0])
        dictionary.append(tmp)
    return dictionary
def create_lsi(num_topic, dictionary):
    corpus, dic= generate_corpus(dictionary)
    print("__________________________Create LSI_________________________")
    lsimodel = LsiModel(corpus=corpus, num_topics=10, id2word=dic)
    topics = lsimodel.print_topics(num_topic)  # Showing only the top 5 topics
    # see list of topics
    for topic in topics:
        print(topic)
    
    return lsimodel

def create_hdp(num_topic, dictionary):
    print("__________________________Create HDP_________________________")
    corpus, dic= generate_corpus(dictionary)
    hdpmodel = HdpModel(corpus=corpus, id2word=dic)
    topics = hdpmodel.print_topics(num_topics=num_topic, num_words= 7)
    # see list of topics
    for topic in topics:
        print(topic)
    
    return hdpmodel

def lda_visualize(ldamodel, dictionary):
    corpus, dic= generate_corpus(dictionary)
    pyLDAvis.enable_notebook()
    vis = pyLDAvis.gensim.prepare(ldamodel, corpus, dic)
    return vis

