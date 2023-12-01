# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 20:54:02 2016

@author: Risto
"""

import os
os.chdir(r'C:\Users\Risto\Documents\Infotehnoloogia mitteinformaatikutele\Programmeerimise alused II\IV nädal\Suurem_ülesanne')

from twython import Twython
twitter = Twython('C8Jpl39xJIVC6uw4zb6Rj809s','VjTKEep9JMmrk8mezvSF3C4igtKkIWg2GObwuJ6TCWB8qsL6q4',
                  '900293220-O76OTSxQzZeB7hvjlwujXYg7rDzN4Dy7bF4LBZn8', 'LGjLVWtiB8gloVC2nhi04KroTlJLtWiic803qiN296JFc')

#korraga saab ühe päringuga 100 tweeti, võtame 1000 ikka
#teeme funktsiooni
#võtame välja teksti
#==============================================================================
# def teksti_eraldaja(tweedid):
#     tekst=[]
#     for i in range(len(tweedid['statuses'])):
#         tekst.append(tweedid['statuses'][i]['text'])
#     #teeme eraldi listidest ühe listi
#     #tulem=''.join(tekst)   
#     return tekst
#==============================================================================
    
import math as math

#==============================================================================
# def allalaadija(märksõna, kogus, keel='en'):
#     tweedid=twitter.search(q=märksõna,lang=keel, count=100)
#     tulem=teksti_eraldaja(tweedid)
#     loopideArv=math.ceil(kogus/100)
#     minid=min_id(tweedid)
#     for i in range(loopideArv-1):
#         uuedtweedid=twitter.search(q=märksõna,lang=keel, count=100, 
#                                    max_id=minid)
#         tulem.extend(teksti_eraldaja(uuedtweedid))
#         minid=min_id(uuedtweedid)
#         print("Tsükkel nr:", str(i+2)+ ", veel tsükleid:",str(loopideArv-i-2))
#     return tulem
#==============================================================================

def teksti_eraldaja(tweedid):
    tekst=[]
    for i in range(len(tweedid)):
        tekst.append(tweedid[i]['text'])  
    return tekst   
    
def allalaadija(märksõna, kogus, keel='en'):
    loopideArv=math.ceil(kogus/100)
    print("Tsükkel nr:", str(1)+ ", veel tsükleid:",str(loopideArv-1))
    tweedid=twitter.search(q=märksõna,lang=keel, count=100)
    tulem=tweedid['statuses']
    minid=min_id(tweedid)
    for i in range(loopideArv-1):
        print("Tsükkel nr:", str(i+2)+ ", veel tsükleid:",str(loopideArv-i-2))
        uuedtweedid=twitter.search(q=märksõna,lang=keel, count=100, 
                                   max_id=minid)
        tulem+=uuedtweedid['statuses']
        minid=min_id(uuedtweedid)
    return tulem
    
estonia=allalaadija('estonia', 2000)

#==============================================================================
# import json
# json.dump(proov, open("proov.txt",'w'))
# proov2=json.load(open("proov.txt"))
# 
# tkst=teksti_eraldaja(proov2)
#==============================================================================

def min_id(tweedid):
    """fun mis leiab allalaaditud tweetide id miinimumi, selle saab
    anda sisendiks järgmisele loobile, et võtab tweedid, mille id on 
    väiksem kui siin leitud id"""
    ids=[]
    for i in range(len(tweedid['statuses'])):
        ids.append(tweedid['statuses'][i]['id'])
    return min(ids)

from nltk.tokenize import RegexpTokenizer
import re
def tokeniseerija(tekstlist):
    tokenizer = RegexpTokenizer(r'\w+')
    tokens=[]
    for i in range(len(tekstlist)):
        #eemaldame kasutajanimed ja lingid, numbrid ja kirjavahemärgid
        abimuutuja=re.sub(r'(^https?:\/\/.*[\r\n]*)|(@[A-Za-z0-9]+)|([0-9])|(\w+:\/\/\S+)', '', tekstlist[i].lower(), flags=re.MULTILINE)
        tokens.append(tokenizer.tokenize(abimuutuja))
    return tokens

from stop_words import get_stop_words
def puhastaja(tokenslist, myralist):
    stopsonad = get_stop_words('en')#inglise keel stoppsõnad
    stopsonad.extend(myralist)#paneme veel juurde
    #eemaldame stoppsõnad ja sõnad alla 2 täheärgi
    tokens_stopped=[]
    for j in range(len(tokenslist)):
        tokens_stopped.append([i for i in tokenslist[j] if not i in stopsonad and len(i)>2])   
    return tokens_stopped

from nltk.stem.porter import PorterStemmer

def stemming(tokens):
    """funktsioon, mis ühtlustab sõnade kuju lõigates pöörde- ja käänel"""
    p_stemmer = PorterStemmer()
    stemmed=[]
    for j in range(len(tokens)):
        stemmed.append([p_stemmer.stem(i) for i in tokens[j]]) 
    return stemmed
    
#estonia=allalaadija('estonia', 2000)
#latvia=allalaadija('latvia', 2000)
#salvestame, et igal korral ei peaks uue päringu tegema
import json
#json.dump(estonia, open("estonia.txt",'w'))
#json.dump(latvia, open("latvia.txt",'w'))
#kui on vaja failist üles laadida:
estonia = json.load(open("estonia.txt"))
latvia=json.load(open("latvia.txt"))
#eraldame teksti
estoniatkst=teksti_eraldaja(estonia)
latviatkst=teksti_eraldaja(latvia)
#tokeniseerime
esttoken=tokeniseerija(estoniatkst)
lattoken=tokeniseerija(latviatkst)
#puhastame
estpuhas=puhastaja(esttoken,['estonia'])
latpuhas=puhastaja(lattoken,['latvia'])
#stemming
eststem=stemming(estpuhas)
latstem=stemming(latpuhas)

###########mudeli ehitamine
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
from gensim.corpora.dictionary import Dictionary
##################Eesti mudel, enne dict tokensitest
estdictionary = Dictionary(eststem)
estcorpus = [estdictionary.doc2bow(text) for text in eststem]
#mudel
estmudel = LdaModel(corpus=estcorpus, id2word=estdictionary, 
                    iterations=500, num_topics=3,  
                    random_state=1)
#topic term
estmudel.get_topic_terms(2)
#perplexity (pole küll korrektne, kuna peaks kasutama houldouti peal)
estmudel.log_perplexity(estcorpus)
#topicud
estmudel.show_topics(3)
estmudel.print_topics(num_topics=3, num_words=3)
#ldavis
import pyLDAvis.gensim
from gensim.models.coherencemodel import CoherenceModel
estldavis=pyLDAvis.gensim.prepare(estmudel, estcorpus, estdictionary) 
#save
pyLDAvis.save_html(estldavis, 'est.html') 

#################läti mudel
latdictionary = Dictionary(latstem)
latcorpus = [latdictionary.doc2bow(text) for text in latstem]
#mudel
latmudel = LdaModel(corpus=latcorpus, id2word=latdictionary, 
                    iterations=500, num_topics=3, random_state=1)
#topic term
latmudel.get_topic_terms(2)
#topicud
latmudel.show_topics(3)
latmudel.print_topics(num_topics=3, num_words=3)
latldavis=pyLDAvis.gensim.prepare(latmudel, latcorpus, latdictionary) 
#save
pyLDAvis.save_html(latldavis, 'lat.html') 


#######################################
##teine variant 
#funktsioon, mis paneb tokensid ühte stringi alamlistis
def sonede_dict(sonedelist):
    token_dict={}
    for i in range(len(sonedelist)):
        token_dict[i] = ' '.join(sonedelist[i])
    return token_dict
    
estdict=sonede_dict(eststem)
latdict=sonede_dict(latstem)
    
from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.feature_extraction import DictVectorizer
print("\n Build DTM")
tfest = CountVectorizer(stop_words='english')
tflat = CountVectorizer(stop_words='english')
print("\n Fit DTM")
tfsest = tfest.fit_transform(estdict.values())
tfslat = tflat.fit_transform(latdict.values())
#get feature names
vocabest = tfest.get_feature_names()
vocablat = tflat.get_feature_names()
# set the number of topics to look for
import lda
n_topics_est=3
n_topics_lat=3
modelest = lda.LDA(n_topics=n_topics_est, n_iter=500, random_state=1)
modellat = lda.LDA(n_topics=n_topics_lat, n_iter=500, random_state=1)

# we fit the DTM not the TFIDF to LDA
print("\n Fit LDA to data set")
%time modelest.fit_transform(tfsest)
%time modellat.fit_transform(tfslat)

print("\n Obtain the words with high probabilities")
%time topic_word_est = modelest.topic_word_  # model.components_ also works
%time topic_word_lat = modellat.topic_word_  # model.components_ also works

#teemad koos sõnadega
import numpy as np

def teema_sona(topic_word, vocab, n_top_words=5):
    for i, topic_dist in enumerate(topic_word):
        topic_words = np.array(vocab)[np.argsort(topic_dist)][:-n_top_words:-1]
        print('Topic {}: {}'.format(i, ' '.join(topic_words)))
  
teema_sona(topic_word_est, vocabest, n_top_words=10)  
teema_sona(topic_word_lat, vocablat,n_top_words=10)  

import matplotlib.pyplot as plt
# use matplotlib style sheet
try:
    plt.style.use('ggplot')
except:
    # version of matplotlib might not be recent
    pass  

#doc_topic = modelest.doc_topic_
doc_topic_est = modelest.doc_topic_
doc_topic_lat = modellat.doc_topic_

#doc_topic_est.shape
#topic document
   
def teema_doku_plot(doc_topic,teemade_arv, doc_n):
    #joonistab konsooli, mitte eraldi
    %matplotlib inline
    f, ax= plt.subplots(len(doc_n), 1, figsize=(8, 6), sharex=True)
    for i, k in enumerate(doc_n):
        ax[i].stem(doc_topic[k,:], linefmt='r-',
               markerfmt='ro', basefmt='w-')
        ax[i].set_xlim(-0.1, teemade_arv-0.9)
        ax[i].set_ylim(0, 1)
        ax[i].set_xticks(np.arange(0, teemade_arv, 1)) 
        ax[i].set_axis_bgcolor((1,0.98,0.98))
        ax[i].set_ylabel("Prob")
        ax[i].set_title("Säuts {}".format(k))
    ax[len(doc_n)-1].set_xlabel("Teema")
    plt.tight_layout()
    plt.show()
    
    
teema_doku_plot(doc_topic_est, n_topics_est, [1,2,3,4])    
teema_doku_plot(doc_topic_lat, n_topics_lat,[14,15,16])    

teema_sona(topic_word_est, vocabest)  
teema_sona(topic_word_lat, vocablat)  

estonia[2]  
latvia[9]
latvia[10]
latstem[9]
latstem[10]

######################################
############################################
#proovin veel ühte mudelit
from sklearn.decomposition import LatentDirichletAllocation
import pyLDAvis.sklearn
lda_est = LatentDirichletAllocation(n_topics=3, random_state=1)
#veelYksmudel=lda_est.fit(tfsest)
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

#modelest.fit_transform(tfsest)
lda_est.fit_transform(tfsest)
#mudeli paramteerid
lda_est.get_params()

#töötab, sõna doku jagunemine
doc_topic_distrib = lda_est.transform(tfsest)
teema_doku_plot(doc_topic_distrib, n_topics_est, [1,2,3,4])

np.amax(doc_topic_distrib,0)
def teema_doku_plot(doc_topic,teemade_arv, doc_n):
    #joonistab konsooli, mitte eraldi
    %matplotlib inline
    f, ax= plt.subplots(len(doc_n), 1, figsize=(8, 6), sharex=True)
    for i, k in enumerate(doc_n):
        #normaliseerin iga rea, kuna muidu läheb üle 1
        ax[i].stem(doc_topic[k,:]/(sum(doc_topic[k,:])),
                 linefmt='r-', markerfmt='ro', basefmt='w-')
        ax[i].set_xlim(-0.1, teemade_arv-0.9)
        ax[i].set_ylim(0, 1)
        ax[i].set_xticks(np.arange(0, teemade_arv, 1)) 
        ax[i].set_axis_bgcolor((1,0.98,0.98))
        ax[i].set_ylabel("Prob")
        ax[i].set_title("Säuts {}".format(k))
    ax[len(doc_n)-1].set_xlabel("Teema")
    plt.tight_layout()
    plt.show()
    
from sklearn.decomposition import NMF, LatentDirichletAllocation
tf_feature_names = tfest.get_feature_names()


def print_top_sonad(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Teema #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()
    
from __future__ import print_function
print_top_sonad(lda_est, tf_feature_names, 10)


ldavisest=pyLDAvis.sklearn.prepare(lda_est, tfsest, tfest)
pyLDAvis.save_html(ldavisest, 'est2.html') 

#läti
lda_lat = LatentDirichletAllocation(n_topics=3, random_state=1)
lda_lat.fit_transform(tfslat)
ldavislat=pyLDAvis.sklearn.prepare(lda_lat, tfslat, tflat)
pyLDAvis.save_html(ldavislat, 'lat2.html') 

# we fit the DTM not the TFIDF to LDA
print("\n Fit LDA to data set")
%time lda_tf.fit_transform(tfsest)

print("\n Obtain the words with high probabilities")
%time topic_word = lda_tf.topic_word_  # model.components_ also works

print("\n Obtain the feature names")
%time vocab = tf.get_feature_names()
  
topic_word = model.topic_word_


