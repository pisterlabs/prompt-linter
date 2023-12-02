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

# gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.models import HdpModel
from gensim.models import TfidfModel

# plotting tools
import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt

# sci-kit
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import feature_extraction



resultFile = open("results.csv",'a')
results = csv.writer(resultFile,lineterminator ='\n')


def main():
    i = 1
    numtops = [5, 8, 10, 15, 20]
    passes = [300]

    total = len(numtops)*len(passes)
    
    for nt in numtops:
        for p in passes:
            test(p, nt, 5)
            print("\n\n\n\n\n\n\n test "+str(i)+" of "+str(total) + " \n\n\n\n\n")
            i += 1

    resultFile.close()
    
def test(passes, topics, trials):

    with open("topicdata\\trial_p"+str(passes)+"_t"+str(topics)+".txt", 'a') as out:

        for t in range(1, trials+1):

            out.write("Trial "+str(t) + "\n")
                    
            lda_model = gensim.models.LdaMulticore(corpus_tfidf, num_topics=topics, id2word=dictionary, passes=passes, workers =4)

            for idx, topic in lda_model.print_topics(-1):
                out.write('Topic: {} Word: {}'.format(idx, topic) + "\n")



# preprocessing
blFile= open("tools\\blacklist.csv", 'r')
blacklist = [t.strip() for t in next(csv.reader(blFile))]
blFile.close()
levels = [1, 2, 3]

# format [term, orig, sentence, docID]
inPath = "raw.csv"

inFile = open(inPath, 'r')
inReader = csv.reader(inFile)

docTokens = dict()

# ignore headers
next(inReader)

for inRow in inReader:
    
    term = inRow[0]
    sentence = inRow[2]
    docID = inRow[3]
    
    # find acceptable tokens only
    token = "_".join([t for t in term.split(":") if re.match(r'[^\W\d]*$', t) and not t in blacklist])
    
    # calculate new term level
    level = token.count("_")
    
    # if acceptable, add to dictionary
    if level in levels and not token in blacklist and len(token) > 0:
        if docID in docTokens:
            docTokens[docID] += [token]
        else:
            docTokens[docID] = [token]

docIDs = list(docTokens.keys())
texts= list(docTokens.values())

dictionary = corpora.Dictionary(texts)

dictionary.filter_extremes(no_below=3, no_above=1, keep_n=10000)
corpus = [dictionary.doc2bow(text) for text in texts]


tfidf = TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]




if __name__ == "__main__":
    main()
    
