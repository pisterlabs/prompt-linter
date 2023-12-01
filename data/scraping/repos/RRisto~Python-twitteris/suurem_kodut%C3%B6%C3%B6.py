# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 16:15:58 2016

@author: Risto
"""
import os
os.chdir(r'C:\Users\Risto\Documents\Infotehnoloogia mitteinformaatikutele\Programmeerimise alused II\IV nädal\Suurem_ülesanne')

from twython import Twython
twitter = Twython('C8Jpl39xJIVC6uw4zb6Rj809s','VjTKEep9JMmrk8mezvSF3C4igtKkIWg2GObwuJ6TCWB8qsL6q4',
                  '900293220-O76OTSxQzZeB7hvjlwujXYg7rDzN4Dy7bF4LBZn8', 'LGjLVWtiB8gloVC2nhi04KroTlJLtWiic803qiN296JFc')


estonia2=twitter.search(q='estonia',lang='en', count=100, max_id=idmax)
latvia=twitter.search(q='russia',lang='en', count=10)
#korraga saab ühe päringuga 100 tweeti, võtame 1000 ikka
#teeme funktsiooni
import math as math

def allalaadija(märksõna, kogus, keel='en'):
    tweedid=twitter.search(q=märksõna,lang=keel, count=100)
    tulem=teksti_eraldaja(tweedid)
    loopideArv=math.ceil(kogus/100)
    minid=min_id(tweedid)
    for i in range(loopideArv-1):
        uuedtweedid=twitter.search(q=märksõna,lang=keel, count=100, 
                                   max_id=minid)
        tulem.extend(teksti_eraldaja(uuedtweedid))
        minid=min_id(uuedtweedid)
        print("Tsükkel nr:", str(i+2)+ ", veel tsükleid:",str(loopideArv-i-2))
    return tulem

def min_id(tweedid):
    """fun mis leiab allalaaditud tweetide id miinimumi, selle saab
    anda sisendiks järgmisele loobile, et võtab tweedid, mille id on 
    väiksem kui siin leitud id"""
    ids=[]
    for i in range(len(estonia2['statuses'])):
        ids.append(estonia2['statuses'][i]['id'])
    return min(ids)

#min_id(estonia)

estonia=allalaadija('estonia', 1000)
latvia=allalaadija('latvia', 1000)
#salvestame, et igal korral ei peaks uue päringu tegema
import json
json.dump(estonia, open("estonia.txt",'w'))
json.dump(latvia, open("latvia.txt",'w'))
#kui on vaja failist üles laadida:
estonia = json.load(open("estonia.txt"))
latvia=json.load(open("latvia.txt"))


 #teeme eraldi listidest ühe listi
tulem=''.join(tekst)   
    return tekst
#1 variant, võtame välja teksti
def teksti_eraldaja(tweedid):
    tekst=[]
    for i in range(len(tweedid['statuses'])):
        tekst.append(tweedid['statuses'][i]['text'])
    #teeme eraldi listidest ühe listi
    #tulem=''.join(tekst)   
    return tekst
    
#esttekst=teksti_eraldaja(estonia)

#raw=''.join(estonia_tekst)

#raw=''.join(twtekst)    

#twtekst=estonia
#twtekstlist=[]
twtekstlist=estonia
#for i in range(len(twtekst)):
#    twtekstlist.append(twtekst[i])
#len(twtekstlist)


from nltk.tokenize import RegexpTokenizer

def tokeniseerija(tekstlist):
    tokenizer = RegexpTokenizer(r'\w+')
    tokens=[]
    for i in range(len(tekstlist)):
        tokens.append(tokenizer.tokenize(tekstlist[i]))
    return tokens

proov=tokeniseerija(estonia)

from stop_words import get_stop_words

def puhastaja(tokensist, myralist):
    stopsonad = get_stop_words('en')#inglise keel stoppsümad
    stopsonad.extend(myra)#paneme veel juurde
    #enna eemaldame numbrid ja teeme tokensid väiketähtedeks
    #eemaldame numbrid ja teeme väiketähtedeks
    tokensLower=[]
    for i in range(len(tokens)):
        tokensLower.append([x.lower() for x in tokens[i] if not any(c.isdigit() for c in x)])
    tokens_stopped=[]
    #eemaldame stoppsõnad
    for j in range(len(tokensLower)):
        tokens_stopped.append([i for i in tokensLower[j] if not i in stopsonad])
    return tokens_stopped

proov2=puhastaja(proov,['co','t', 'http','o','rt','https', 'estonia'])
#stemming
from nltk.stem.porter import PorterStemmer

def stemming(tokens):
    p_stemmer = PorterStemmer()
    stemmed=[]
    for j in range(len(tokens)):
        stemmed.append([p_stemmer.stem(i) for i in tokens[j]]) 
    return stemmed

proov3=stemming(proov2)
#raw = raw.lower()
#tokens = tokenizer.tokenize(raw)
#tokens=[]
#for i in range(len(twtekstlist)):
#    tokens.append(tokenizer.tokenize(twtekstlist[i]))
#len(tokens)         

tokens=[]
for i in range(len(estonia)):
    tokens.append(tokenizer.tokenize(twtekstlist[i]))
len(tokens)     

from stop_words import get_stop_words
# inglise keele stopsõnad
stopsonad = get_stop_words('en')
myra=['co','t', 'http','o','s','rt','https', 'estonia']
stopsonad.extend(myra) #psaneme oma sõnad juurde
#eemaldame numbrid ja teeme väiketähtedeks
tokensLower=[]
for i in range(len(tokens)):
    tokensLower.append([x.lower() for x in tokens[i] if not any(c.isdigit() for c in x)])
len(tokensLower)
# remove stop words from tokens
#stopped_tokens = [i for i in tokens if not i in en_stop]
tokens_stopped=[]
for j in range(len(tokensLower)):
    tokens_stopped.append([i for i in tokensLower[j] if not i in stopsonad])
len(tokens_stopped)

#stemming
from nltk.stem.porter import PorterStemmer
p_stemmer = PorterStemmer()

texts=[]
for j in range(len(tokens_stopped)):
    texts.append([p_stemmer.stem(i) for i in tokens_stopped[j]])


#texts = [p_stemmer.stem(i) for i in stopped_tokens]
#texts=[]
#for j in range(len(tokens)):
 #   texts.append([p_stemmer.stem(i) for i in tokens[j]])

#texts_clean=[]
#myra=['co','t', 'http','o','rt', 'RT','https', '[0-9]', 'Estonia']
#puhas=[]
#for j in range(len(texts)):
#    puhas.append([i for i in tokens[j] if not i in myra])
#print(puhas[0])
 
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
from gensim.corpora.dictionary import Dictionary
dictionary = Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

mudel = LdaModel(corpus=corpus, id2word=dictionary, iterations=500,
                 num_topics=5, alpha=0.1)
 
proov=mudel.get_topic_terms(2)
proov=mudel.show_topics()


    
import pyLDAvis.gensim
from gensim.models.coherencemodel import CoherenceModel
proov=pyLDAvis.gensim.prepare(mudel, corpus, dictionary) 
#save
pyLDAvis.save_html(proov, 'poov.html') 

teemad=mudel.print_topics(num_topics=3, num_words=3)
teemad


##teine variant mudelist
token_dict = {}

for i in range(len(esttekst)):
    token_dict[i] = esttekst[i]


#for i in range(len(twtekst)):
#    token_dict[i] = twtekst[i]

from sklearn.feature_extraction.text import CountVectorizer

#from sklearn.feature_extraction import DictVectorizer
print("\n Build DTM")
%time tf = CountVectorizer(stop_words='english')

print("\n Fit DTM")
%time tfs1 = tf.fit_transform(token_dict.values())

#tfs1.get_feature_names()
  
# set the number of topics to look for
num = 5
import lda
model = lda.LDA(n_topics=num, n_iter=500, random_state=1)




# we fit the DTM not the TFIDF to LDA
print("\n Fit LDA to data set")
%time model.fit_transform(tfs1)

print("\n Obtain the words with high probabilities")
%time topic_word = model.topic_word_  # model.components_ also works

print("\n Obtain the feature names")
%time vocab = tf.get_feature_names()
  
topic_word = model.topic_word_

import matplotlib.pyplot as plt

# use matplotlib style sheet
try:
    plt.style.use('ggplot')
except:
    # version of matplotlib might not be recent
    pass  

%matplotlib inline
f, ax= plt.subplots(5, 1, figsize=(8, 6), sharex=True)
for i, k in enumerate([0,1,2,3,4]):
    ax[i].stem(topic_word[k,:], linefmt='b-',
               markerfmt='bo', basefmt='w-')
    ax[i].set_xlim(-5,10)
    ax[i].set_ylim(0, 0.08)
    ax[i].set_ylabel("Prob")
    ax[i].set_title("topic {}".format(k))

ax[4].set_xlabel("word")

plt.tight_layout()
plt.show()

doc_topic = model.doc_topic_
doc_topic.shape

#topic document
f, ax= plt.subplots(5, 1, figsize=(8, 6), sharex=True)
for i, k in enumerate([0, 5, 9, 14, 19]):
    ax[i].stem(doc_topic[k,:], linefmt='r-',
               markerfmt='ro', basefmt='w-')
    ax[i].set_xlim(-1, 5)
    ax[i].set_ylim(0, 1)
    #ax[i].set_axis_bgcolor('white')
    ax[i].set_axis_bgcolor((1,0.98,0.98))
    ax[i].set_ylabel("Prob")
    ax[i].set_title("Säuts {}".format(k))

ax[4].set_xlabel("Teema")

plt.tight_layout()
plt.show()

############################################
#proovin veel ühte mudelit
from sklearn.decomposition import LatentDirichletAllocation
import pyLDAvis.sklearn
lda_tf = LatentDirichletAllocation(n_topics=5, random_state=1)
veelYksmudel=lda_tf.fit(tfs1)

lda_tf.fit_transform(tfs1)

proov=pyLDAvis.sklearn.prepare(lda_tf, tfs1, tf)
pyLDAvis.save_html(proov, 'poov2.html') 

# we fit the DTM not the TFIDF to LDA
print("\n Fit LDA to data set")
%time lda_tf.fit_transform(tfs1)

print("\n Obtain the words with high probabilities")
%time topic_word = lda_tf.topic_word_  # model.components_ also works

print("\n Obtain the feature names")
%time vocab = tf.get_feature_names()
  
topic_word = model.topic_word_

import matplotlib.pyplot as plt



import numpy as np
import logging
import json
import warnings
warnings.filterwarnings('ignore')  # To ignore all warnings that arise here to enhance clarity



from numpy import array