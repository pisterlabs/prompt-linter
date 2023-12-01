# processing
import operator
from operator import methodcaller
import csv
import re
import numpy as np
import pandas as pd
from pprint import pprint
import string
import math
import itertools
import sqlite3
import copy

# gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.models import HdpModel
from gensim.models import TfidfModel

# nltk
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer

# plotting tools
import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 


## LOAD R&R


# path to blacklisted tokens
blacklist = [t.strip() for t in next(csv.reader(open("tools\\blacklist.csv", 'r')))]

# levels of R&R terms considered
levels = [1, 2, 3]

# format [term, orig, sentence, docID]
inPath = "raw.csv"
inFile = open(inPath, 'r')
inReader = csv.reader(inFile)


docTokens = dict()


next(inReader)
for inRow in inReader:
    term = inRow[0]
    sentence = inRow[2]
    docID = inRow[3]
    
    token = "_".join([t for t in term.split(":") if re.match(r'[^\W\d]*$', t) and not t in blacklist])
    
    level = token.count("_")
    
    if level in levels and not token in blacklist and len(token) > 0:
        print(token)
        if docID in docTokens:
            docTokens[docID] += [token]
        else:
            docTokens[docID] = [token]

docIDs = list(docTokens.keys())
texts = list(docTokens.values())

dictionary = corpora.Dictionary(texts)
print(len(dictionary))

dictionary.filter_extremes(no_below=1, no_above=.25, keep_n=15000)
print(len(dictionary))

corpus = [dictionary.doc2bow(text) for text in texts]


tfidf = TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]



# LOAD UNIGRAMs


stop = set(stopwords.words('english'))
exclude = set(string.punctuation) .union(string.digits)
lemma = WordNetLemmatizer()
def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop and i not in blacklist])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized



# format [term, orig, sentence, docID]
inPath = "v1.csv"
inFile = open(inPath, 'r')
inReader = csv.reader(inFile)


docTokens = dict()

next(inReader)

for inRow in inReader:
    term = inRow[0]
    sentence = inRow[2]
    docID = inRow[3]
    
    if docID in docTokens:
        if sentence not in docTokens[docID]:
            docTokens[docID] += sentence
    else:
        docTokens[docID] = sentence

rawtext = docTokens.values()

testTexts = [clean(doc).split() for doc in rawtext] 
# build a dictionary for the text
testDict = corpora.Dictionary(testTexts)

# filter out terms that appear in fewer that LOW docs or greater than HIGH percent of docs and use KEEP terms
LOW = 1
HIGH = 0.25
KEEP = 17000


testDict.filter_extremes(no_below = LOW, no_above = HIGH, keep_n = KEEP)


# convert the text to a corpus with the dictionary
testCorpus = [testDict.doc2bow(text) for text in testTexts]





# DEFINE TESTERS


def testRR(passes):
    
    model = gensim.models.LdaMulticore(corpus, 
                                       num_topics = 6,
                                       id2word=dictionary,
                                       passes=passes,
                                       workers = 4)
    
    coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_uci')
    
    return coherencemodel.get_coherence()

def testUni(passes):
    
    model = gensim.models.LdaMulticore(testCorpus, 
                                       num_topics = 6,
                                       id2word=testDict,
                                       passes=passes,
                                       workers = 4)
        

    coherencemodel = CoherenceModel(model=model, texts=testTexts, dictionary=testDict, coherence='c_uci')
    
    return coherencemodel.get_coherence()
    
    




# TESTS




passes = [5, 8, 12, 20, 25]
coh1 = list()
coh2 = list()

for p in passes:
    
    coh1.append(testRR(p))
    coh2.append(testUni(p))


plt.plot(passes, coh1, 'r', label = "R&R")
plt.plot(passes, coh2, 'b', label = "Unigram")
plt.xlabel("Passes")
plt.ylabel("Coherence score")
plt.legend()
plt.show()

