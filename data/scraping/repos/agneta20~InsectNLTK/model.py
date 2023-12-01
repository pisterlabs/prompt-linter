'''
Created on Sep 20, 2017

Module created to apply topic modelling. This is the main module that performs lDA and HDP. The module takes data
from MongoDB, gets the text, and then applies that text to the topic model which is then outputed in csv term and topic outputs 
and html files used to visualize the LDA topic model results using different topic numbers.

@author: 
'''

import os
import re
import nltk

import operator
#import matplotlib.pyplot as plt
import warnings
import gensim
import numpy as np

import sys
import csv
from nltk.tokenize import RegexpTokenizer
import pyLDAvis.gensim

from gensim.models import CoherenceModel, LdaModel, LsiModel, HdpModel
from gensim.models.wrappers import LdaMallet
from gensim.corpora import Dictionary
from pprint import pprint
from gensim.utils import lemmatize
from nltk.corpus import stopwords
import pyLDAvis.gensim

from patternMatching import PatternMatcher
import db.filterBeetle 
import db.filterGovernment

import wordninja
from db import filterBeetle
from pattern.db import year

import logging

# Let's not pay heed to them right now
warnings.filterwarnings('ignore')  

# nltk stopwords list
stopwords=stopwords.words('english')
stops = set(stopwords)  

# list of results to hold
listResults=[] 

# results with dates 
dateResults={} 

# results based on government document types
typeResults={} 

#results based on beetles and relevant sentences
beetleResults={} 

#texts used in training or application
train_texts=''

def integrateText(content):
    '''
    Method used to retrieve text, which comes from the database, and integrate together for larger analysis in topic
    models.
    content-- the content with the data to retrieve.
    '''
    txt=''
    
    cnt=[]
    
#    gLevel=["Gov Level 1"]
#    flt=["all"]
    
#    content=db.filterGovernment.filterGovernmentType(flt,gLevel, content)
    
    
    for c in content:
        
        text=c.get('Text',"")
#       text=fb.filterBeetleSentences(text)
        year=c.get('Year')
        
        text=text.strip()
        
        txt=txt+" "+text
    
#        cc={}
#        cc['Text']=txt
#        cc['Year']=year
        
        cnt.append(text)
    
 #   cnt=fb.filterBeetleSentences(cnt)
    
    return cnt


def retrieveText(content):
    '''
    Method to get the text output from results in a CSV. Retrieves relevant texts only.
    content-- the content to retrieve text
    '''
    del listResults[:]
        
    doc_set=[]
    
#    txt=''
    
    iit=0
    nwText=''
    for c in content:
        
        text=c['Text']
        year=c['Year']
        
        text=text.strip()
                            
        text=re.sub('"',' ',text)
        text=re.sub(',',' ',text)
        
#        lsWrds=wordninja.split(text)
                             
#        tokenizer = RegexpTokenizer(r'\w+')
       
        text = nltk.word_tokenize(text)
        
#       nText=[]
        
        newttL=[]
        for t in text:
            
            ts=wordninja.split(t)
            
            
            newtt=''
            for tt in ts:
                newtt+=tt+" "
             
                
            newttL.append(newtt)
            
        
        for nn in newttL:
            nwText+=nn+" "
        
       
        
        del text[:]
#       stopped_tokens = [t for t in nText if not t in stops]
                              
#       doc_set.append(stopped_tokens)  
        
        print(iit) 
        
        print(len(doc_set))
        iit+=1 
        
#       docResults={year:stopped_tokens}  

    doc_set.append(nwText)        
    return doc_set

def test_directories():
    '''
    Data files loaded from test. It returns a file for training or testing.
    '''

    test_data_dir = '{}'.format(os.sep).join([gensim.__path__[0], 'test', 'test_data'])
    lee_train_file = test_data_dir + os.sep + 'lee_background.cor'
    
    with open(lee_train_file) as f:
        for n, l in enumerate(f):
            if n < 5:
                print([l])

    return lee_train_file
'''
def build_texts(fname):
    """
    Function to build tokenized texts from file
    
    Parameters:
    ----------
    fname-- File to be read
    
    Returns:
    -------
    yields preprocessed line
    """
    with open(fname) as f:
        for line in f:
            yield gensim.utils.simple_preprocess(line, deacc=True, min_len=3)
'''

def preProcsText(files):
    '''
    Another method to process text and tokenize files based on minimum length of file

    files-- text for processing
    '''
   
    for f in files:
        f=yield gensim.utils.simple_preprocess(f, deacc=True, min_len=3)
            
    


def process_texts(bigram, texts):
    """
    Function to process texts. Following are the steps we take:
    
    1. Stopword Removal.
    2. Collocation detection.
    3. Lemmatization (not stem since stemming can reduce the interpretability).
    
    Parameters:
    ----------
    bigram-- bigram to analyze
    texts-- Tokenized texts.
    
    Returns:
    -------
    texts: Pre-processed tokenized texts.
    """
    
    # reg. expression tokenizer
    
    texts = [[word for word in line if word not in stops] for line in texts]
    texts = [bigram[line] for line in texts]
    texts = [[word.split('/')[0] for word in lemmatize(' '.join(line), allowed_tags=re.compile('(NN)'), min_length=3)] for line in texts]

    return texts


def evaluate_graph(dictionary, corpus, texts, limit):
    """
    Method for using a coherence model to look at topic coherence for LDA models.
    
    Parameters:
    ----------
    dictionary-- Gensim dictionary
    corpus-- Gensim corpus
    limit-- topic limit
    
    Returns:
    -------
    lm_list : List of LDA topic models
    c_v : Coherence values corresponding to the LDA model with respective number of topics
    """
    c_v = []
    lm_list = []
    for num_topics in range(1, limit):
        lm = LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary)
        lm_list.append(lm)
        cm = CoherenceModel(model=lm, texts=texts, dictionary=dictionary, coherence='c_v')
        c_v.append(cm.get_coherence())
        del cm
            
    return lm_list, c_v


def ret_top_model(corpus,dictionary):
    """
    Since LDAmodel is a probabilistic model, it comes up different topics each time we run it. To control the
    quality of the topic model we produce, we can see what the interpretability of the best topic is and keep
    evaluating the topic model until this threshold is crossed. 
    corpus-- the text corpus
    dictionary-- term dictionary

    method returns lm: final evaluated topic model
    method returns top_topics: ranked topics in decreasing order. List of tuples
    """
    top_topics = [(0, 0)]
    while top_topics[0][1] < 0.97:
        lm = LdaModel(corpus=corpus, id2word=dictionary)
        coherence_values = {}
        for n, topic in lm.show_topics(num_topics=-1, formatted=False):
            topic = [word for word, _ in topic]
            cm = CoherenceModel(topics=[topic], texts=train_texts, dictionary=dictionary, window_size=10)
            coherence_values[n] = cm.get_coherence()
        top_topics = sorted(coherence_values.items(), key=operator.itemgetter(1), reverse=True)
    return lm, top_topics


def addTotalTermResults(t):
    '''Method to add results and clean by removing extra white space
    t-- results and text to clean
   
    result_dict: dictionary of the term and values that are cleaned
    '''
    result_dict={}
    for a,b in t:
            text=re.sub('"',"",b)
            text.replace(" ","")
            txts=text.split("+")
            for t in txts:
                ttnts=t.split("*")
                v=float(ttnts[0])
                t=ttnts[1]
                t=str(a)+":"+t
                if(t in result_dict):
                    continue
                else:
                    t=t.strip()
                    result_dict[t]=v 
                           
    return result_dict
                        
       
def addToResults(result_dict):
    '''
    Add dictionary to a list of results from each text
    result_dict-- this is the resulting terms
    ''' 
    listResults.append(result_dict)
            
                  
def dictionaryResults():
    
    '''
    Method aggregates all the dictionaries for keyterms and their values.
    returns a dictionary of all keyterms and values
    ''' 
    #set the dictionary
    dct={}
        
    #iterate over all tweets and add to total dictionary
    for dictionary in listResults:
            for key in dictionary:
                    
                v=dictionary[key]
                if(key in dct):
                    vv=dct[key]
                    vv=v+vv
                    dct[key]=vv
                else:
                    dct[key]=v 
                        
    return dct
    

def printResults(i,model):
        '''Output results of the analysis
        i-- the topic number
        model-- the model used (e.g., lda, hdp)
        '''
        #os.chdir('../')
        pn=os.path.abspath(__file__)
        pn=pn.split("src")[0]
        filename=os.path.join(pn,'results','analysis_results_'+model+str(i)+".csv")
        
        fieldnames = ['Topic','Term','Value']
        
        dct=dictionaryResults()
        with open(filename, 'wb') as csvf:
            writer = csv.DictWriter(csvf, fieldnames=fieldnames)

            writer.writeheader()
            
            for key in dct:
                v=dct[key]
                tn=key.split(":")[0]
                kt=key.split(":")[1]
                writer.writerow({'Topic':str(tn),'Term': str(kt.encode("utf-8")),'Value':str(v)})
        
def printEvaluation(modList,results,i):
    
    '''
    Method to print csv output results of the evaluations conducted 

    modList-- the model evaluated
    results-- the result scores
    i-- the index output desired
    '''
    pn=os.path.abspath(__file__)
    pn=pn.split("src")[0]
    filename=os.path.join(pn,'results','evaluationTotal'+str(i)+".csv")   
        
    fieldnames = ['Model','Score']
    
    with open(filename, 'wb') as csvf:
            writer = csv.DictWriter(csvf, fieldnames=fieldnames)

            writer.writeheader()
            for i in range(0,len(modList)):
        
                writer.writerow({'Model':str(modList[i]),'Score': str(results[i])})
        
#lee_train_file=test_directories()
#train_texts = list(build_texts(lee_train_file))

#bigram = gensim.models.Phrases(train_texts)

def run():
    '''
    The method to run for implementing the topic models (LDA and HDP).
    The below code is executed to conduct the models for topic modelling and 
    coherence testing for LDA models
    '''
    topicN=raw_input("Number of topics:  ")
    fll=raw_input("Filter Terms: ")
    gov=raw_input("Government Material: ")
    fflt=fll.split(",")
    
    flt=[]
    for f in fflt:
        flt.append(f)
        
    ##filter based on government type
    ##filter based on sentences around mountain pine beetle
    ##do one filter at a time and then both together
           
    pn=os.path.abspath(__file__)
    pn=pn.split("src")[0]

    p=PatternMatcher()
   

    content=p.retrieveContent(flt,gov)
    results=integrateText(content)

    #results=retrieveText(results)

    bigram = gensim.models.Phrases(results) 
    #train_texts = process_texts(results)

    results=preProcsText(results)

    train_texts=process_texts(bigram,results)

    print('start')

    dictionary = Dictionary(train_texts)
    corpus = [dictionary.doc2bow(text) for text in train_texts]

    #keep track of iteration
    iiT=2

    #topics are tested based on a given topic number
    for i in range(2,int(topicN),1): 
#    lsi model
    
        print('run evaluation: '+ str(i))
    
        #lsimodel = LsiModel(corpus=corpus, num_topics=i, id2word=dictionary)

        #lsitopics=lsimodel.show_topics(num_topics=i)

        #result_dict=addTotalTermResults(lsitopics)    
        #addToResults(result_dict)
        #printResults(i,'lsi')
    
        del listResults[:]    
    
        #hdp model
        hdpmodel = HdpModel(corpus=corpus, id2word=dictionary)

        hdpmodel.show_topics()

        hdptopics = hdpmodel.show_topics(num_topics=i)

        result_dict=addTotalTermResults(hdptopics)
            
        #add results to total kept in a list     
        addToResults(result_dict)
    
        printResults(i,'hdp')
        del listResults[:] 
     
        #lda model
        ldamodel = LdaModel(corpus=corpus, num_topics=i, id2word=dictionary)
        num=str(i)
        ldamodel.save('lda'+num+'.model')
        ldatopics = ldamodel.show_topics(num_topics=i)
    
        result_dict=addTotalTermResults(ldatopics)    
        addToResults(result_dict)
        printResults(i,'lda')
    
        del listResults[:] 
    
        visualisation2 = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)
   
        location=os.path.join(pn,'results')
     
        #visualize outputs
        pyLDAvis.save_html(visualisation2, os.path.join(location,'LDA_Visualization'+str(i)+'.html')) 
    
    
    iiT=i

    print('evaluate graph')

    #coherence model evaluation
    lmlist, c_v = evaluate_graph(dictionary=dictionary, corpus=corpus, texts=train_texts, limit=i)

    #lm, top_topics = ret_top_model()

    #coherence model results
    printEvaluation(lmlist,c_v,iiT)
     
''''
The main to launch this class for merging lda and hdp terms and topics
'''
if __name__ == '__main__':
    run() 
