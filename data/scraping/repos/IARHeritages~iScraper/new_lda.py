'''
Created on Sep 20, 2017

@author: maltaweel
'''
from lda import LDA
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import gensim
from gensim import corpora
Lda = gensim.models.ldamodel.LdaModel
import os
import csv
import re
import sys
import gensim
import numpy as np
import operator
import matplotlib.pyplot as plt
import warnings

from gensim.models import CoherenceModel, LdaModel, LsiModel, HdpModel
from gensim.models.wrappers import LdaMallet
from gensim.corpora import Dictionary
from pprint import pprint


stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()


class NEW_LDA(LDA):
    
    '''
    Method to get the text output from the scraping.
    @param pn the path to find the relevant text
    '''
    def retrieveText(self,pn):
        del self.listResults[:]
        
        doc_set=[]
        os.chdir(pn+'/test')
        
        result=[]
        for filename in os.listdir(os.getcwd()):
            txt=''
            if(filename == ".DS_Store" or "lda" in filename or "hdp" in filename or ".csv" not in filename):
                continue
            print(filename)
            with open(filename, 'rU') as csvfile:
                reader = csv.reader(csvfile, quotechar='|') 
                
                i=0
                try:
                    for row in reader:
                        if row in ['\n', '\r\n']:
                            continue;
                        if(i==0):
                            i=i+1
                            continue
                        if(len(row)<1):
                            continue
                        
                        text=''
                        for r in row:
                            text+=r
                        text=re.sub('"','',text)
                        text=re.sub(',','',text)
                    
                        tFalse=True
                        
                        if(len(result)==0):
                            result.append(text)
                            i+=1
                            txt=txt+" "+text
     #                       continue
     #                   for s in result:
     #                       if(text in s):
     #                           tFalse=False
     #                           break
                            
                        if(tFalse==True):
     #                       result.append(text)
                             txt=txt+" "+text
                             doc_set.append(unicode(text, errors='replace'))  
                        i+=1 
                except csv.Error, e:
                    sys.exit('line %d: %s' % (reader.line_num, e))
            
               
     #           doc_set.append(unicode(txt, errors='replace'))
            
        return doc_set
    
    def clean(self,doc):
        stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
        punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
        normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
        return normalized
    

    def runClean(self,doc_complete):
        doc_clean = [self.clean(doc).split() for doc in doc_complete] 
        
        return doc_clean
    
    def doc_term_prep(self,doc_clean):
        self.dictionary = corpora.Dictionary(doc_clean)
        doc_term_matrix = [self.dictionary.doc2bow(doc) for doc in doc_clean]
        
        return doc_term_matrix
        
    
    def runModel(self,doc_term_matrix,num_topics,num_words,passes):
        ldamodel = Lda(doc_term_matrix, num_topics=num_topics, id2word = self.dictionary, passes=passes)
        t=ldamodel.print_topics(num_topics=num_topics, num_words=num_words)
            
            
        #term and values from text
        result_dict=self.addTotalTermResults(t)
            
        #add results to total kept in a list     
        self.addToResults(result_dict)

pn=os.path.abspath(__file__)
pn=pn.split("src")[0]

nl=NEW_LDA()
doc_set=nl.retrieveText(pn)
doc_clean=nl.runClean(doc_set)
doc_term_matrix = nl.doc_term_prep(doc_clean)

nl.runModel(doc_term_matrix,20,20,20)
nl.printResults(20, 20, 20)




