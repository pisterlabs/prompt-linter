'''
Created on Sep 20, 2017

@author: maltaweel
'''


import os
import re
from hdp import HDP
import operator
#import matplotlib.pyplot as plt
import warnings
import gensim
import numpy as np
warnings.filterwarnings('ignore')  # Let's not pay heed to them right now
import csv
from nltk.tokenize import RegexpTokenizer

from gensim.models import CoherenceModel, LdaModel, LsiModel, HdpModel
from gensim.models.wrappers import LdaMallet
from gensim.corpora import Dictionary
from pprint import pprint
from gensim.utils import lemmatize
from nltk.corpus import stopwords
import pyLDAvis.gensim

stops = set(stopwords.words('english'))  # nltk stopwords list
listResults=[]
def test_directories():
    test_data_dir = '{}'.format(os.sep).join([gensim.__path__[0], 'test', 'test_data'])
    lee_train_file = test_data_dir + os.sep + 'lee_background.cor'
    
    with open(lee_train_file) as f:
        for n, l in enumerate(f):
            if n < 5:
                print([l])

    return lee_train_file

def build_texts(fname):
    """
    Function to build tokenized texts from file
    
    Parameters:
    ----------
    fname: File to be read
    
    Returns:
    -------
    yields preprocessed line
    """
    with open(fname) as f:
        for line in f:
            yield gensim.utils.simple_preprocess(line, deacc=True, min_len=3)
            
def preProcsText(files):
  
        for f in files:
            yield gensim.utils.simple_preprocess(f, deacc=True, min_len=3)

def process_texts(texts):
    """
    Function to process texts. Following are the steps we take:
    
    1. Stopword Removal.
    2. Collocation detection.
    3. Lemmatization (not stem since stemming can reduce the interpretability).
    
    Parameters:
    ----------
    texts: Tokenized texts.
    
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
    Function to display num_topics - LDA graph using c_v coherence
    
    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    limit : topic limit
    
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
        
    # Show graph
#    x = range(1, limit)
#    plt.plot(x, c_v)
#    plt.xlabel("num_topics")
#    plt.ylabel("Coherence score")
#    plt.legend(("c_v"), loc='best')
#    plt.show()
    
    return lm_list, c_v

def ret_top_model():
    """
    Since LDAmodel is a probabilistic model, it comes up different topics each time we run it. To control the
    quality of the topic model we produce, we can see what the interpretability of the best topic is and keep
    evaluating the topic model until this threshold is crossed. 
    
    Returns:
    -------
    lm: Final evaluated topic model
    top_topics: ranked topics in decreasing order. List of tuples
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

    '''The terms and values from text.
    @return result_dict dictionary of the term and values'''
def addTotalTermResults(t):
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
                        
'''Add dictionary to a list of results from each text
    @param result_dict this is the resulting terms'''        
def addToResults(result_dict):
        listResults.append(result_dict)
            
        
'''Method aggregates all the dictionaries for keyterms and their values.
    @return dct a dictionary of all keyterms and values'''           
def dictionaryResults():
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
    
'''Output results of the analysis
@param nn the number of topics used for the output name
@param i topic number
@param model the model
'''
def printResults(i,model):
        
  #     os.chdir('../')
        pn=os.path.abspath(__file__)
        pn=pn.split("src")[0]+'/'+model
        
        filename=pn+'/'+model+'_results'+"-"+str(i)+'-'+'.csv'
        
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
        

#lee_train_file=test_directories()
#train_texts = list(build_texts(lee_train_file))

#bigram = gensim.models.Phrases(train_texts)

hdp=HDP() 
pn=os.path.abspath(__file__)
pn=pn.split("src")[0]
results=hdp.retrieveText(pn)

bigram = gensim.models.Phrases(results) 
#train_texts = process_texts(train_texts)

train_texts=process_texts(results)


preProcsText(results)

dictionary = Dictionary(train_texts)
corpus = [dictionary.doc2bow(text) for text in train_texts]

for i in range(10,100,10):
    lsimodel = LsiModel(corpus=corpus, num_topics=i, id2word=dictionary)

    lsitopics=lsimodel.show_topics(num_topics=i)

    result_dict=addTotalTermResults(lsitopics)    
    addToResults(result_dict)
    printResults(i,'lsi')
    
    del listResults[:]    
    hdpmodel = HdpModel(corpus=corpus, id2word=dictionary)

    hdpmodel.show_topics()

    hdptopics = hdpmodel.show_topics(num_topics=i)

    result_dict=addTotalTermResults(hdptopics)
            
    #add results to total kept in a list     
    addToResults(result_dict)
    
    printResults(i,'hdp')
    del listResults[:] 
     
    ldamodel = LdaModel(corpus=corpus, num_topics=i, id2word=dictionary)

    ldatopics = ldamodel.show_topics(num_topics=i)
    
    result_dict=addTotalTermResults(ldatopics)    
    addToResults(result_dict)
    printResults(i,'lda')
    del listResults[:] 
    
    lmlist, c_v = evaluate_graph(dictionary=dictionary, corpus=corpus, texts=train_texts, limit=10)

    #lm, top_topics = ret_top_model()
    lmtopics = lmlist[5].show_topics(formatted=False)

    #print(top_topics[:5])

#print([lm.show_topic(topicid) for topicid, c_v in top_topics[:10]])

#lda_lsi_topics = [[word for word, prob in lm.show_topic(topicid)] for topicid, c_v in top_topics]

#lsitopics = [[word for word, prob in topic] for topicid, topic in lsitopics]

#hdptopics = [[word for word, prob in topic] for topicid, topic in hdptopics]

#ldatopics = [[word for word, prob in topic] for topicid, topic in ldatopics]

#lmtopics = [[word for word, prob in topic] for topicid, topic in lmtopics]

#lsi_coherence = CoherenceModel(topics=lsitopics[:10], texts=train_texts, dictionary=dictionary, window_size=10).get_coherence()

#hdp_coherence = CoherenceModel(topics=hdptopics[:10], texts=train_texts, dictionary=dictionary, window_size=10).get_coherence()

#lda_coherence = CoherenceModel(topics=ldatopics, texts=train_texts, dictionary=dictionary, window_size=10).get_coherence()

#lm_coherence = CoherenceModel(topics=lmtopics, texts=train_texts, dictionary=dictionary, window_size=10).get_coherence()

#lda_lsi_coherence = CoherenceModel(topics=lda_lsi_topics[:10], texts=train_texts, dictionary=dictionary, window_size=10).get_coherence()

