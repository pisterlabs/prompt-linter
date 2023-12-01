import numpy as np
from sklearn import feature_extraction
from sklearn.feature_extraction.text import CountVectorizer
from pyevolve import G1DList
from pyevolve import GSimpleGA
from pyevolve import Crossovers
import numpy as np
import os
import sys
from gensim import corpora, models, interfaces
import gensim
from itertools import izip
from joblib import Parallel, delayed
import multiprocessing
from multiprocessing import Process, Manager
from threading import Thread
import scipy.spatial
from gensim.models.coherencemodel import CoherenceModel



# foldermain = 'RQ3'
foldername = sys.argv[1]
foldermodels = sys.argv[2]
print foldername, foldermodels

clu2orig={}
docTopicProbMat=None
corpus = []
fileList = os.listdir(foldername)
count = 0
corpus = []
texts = []
rc = 0
for f in fileList:
    if (rc % 10000 == 0):
        print("Processed ::" + str(rc) + ":: Files ")
    f = open(foldername + f, 'r')
    txt = str(f.read())
    corpus.append(txt)
    texts.append(txt.split())
    rc += 1

dictionary = corpora.Dictionary(texts)
# dictionary.filter_extremes(no_below=1000, no_above=0.5)
corpus2 = [dictionary.doc2bow(text) for text in texts]
dictionary.save(foldermodels+'MultiCore.dict')
corpora.MmCorpus.serialize(foldermodels+'MultiCoreCorpus.mm', corpus2)

# term frequency
NumApp = len(corpus)
NumFeatures = len(dictionary)
#vectorizer=CountVectorizer(stop_words='english', strip_accents='ascii', max_features=NumFeatures, dtype=np.int32)
# vectorizer = CountVectorizer(max_features=NumFeatures, dtype=np.int32)
# tf_array = vectorizer.fit_transform(corpus).toarray()
# vocab = vectorizer.get_feature_names()


print("Starting Mutations::")
print(NumApp)

print(NumFeatures)
# NumFeatures = len(vocab)
# print(NumFeatures)
print(count)
Centers = []
Clusters = []
classes = []
logfile=open(foldermodels+'/log.txt','w')
# sqD=scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(tf_array))
shScore = {}
coScore = {}

topic_num = None

def eval_coherence(NTopic):
    # print NTopic[0]
    numoftopics = int((8*NTopic[0] + 500) / 330)
    iters = NTopic[1]
    al = (float(NTopic[2]) - 20) / 19800
    bet = (float(NTopic[3]) - 20) / 19800
    if al==0.0:
        al = 1/480
    if bet==0.0:
        bet = 1/480
    global coScore
    log=str(numoftopics) + ' ' + str(iters) + ' ' + str(al) + ' ' + str(bet)
    print log
    if not log in coScore:
        logfile.write(log + "\n")
        model = gensim.models.ldamulticore.LdaMulticore(corpus2,passes=10, num_topics=numoftopics, id2word=dictionary,
                                                        iterations=iters, alpha=al, eta=bet, random_state=123)
        cm = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coScore[log] = cm.get_coherence()
        logfile.write("SCORE::" + str(coScore[log]) + "\n")
    return coScore[log]

def eval_func_JustModel(NTopic):
    global count
    # NTopic[0]=2
    print NTopic[0]
    global topic_num
    numoftopics = int((8*NTopic[0] + 500) / 330)
    topic_num = numoftopics
    iters = NTopic[1]
    al = (float(NTopic[2]) - 20) / 19800
    bet = (float(NTopic[3]) - 20) / 19800

    log=str(count)+' '+str(numoftopics) + ' ' + str(iters) + ' ' + str(al) + ' ' + str(bet)
    print log
    logfile.write(log + "\n")
    print("Creating Model::" + str(count))
    model = gensim.models.ldamulticore.LdaMulticore(corpus2,passes=10, num_topics=numoftopics, id2word=dictionary,iterations=iters,alpha=al,eta=bet,random_state=123)
    cm = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
    print "Coherence score: " + str(cm.get_coherence())
    model.save(foldermodels+str(numoftopics) +'_'+str(iters) + '.model')
    doc_topic_list = []


genome = G1DList.G1DList(4)
genome.evaluator.set(eval_coherence)
genome.setParams(rangemin=20, rangemax=2000)
genome.crossover.set(Crossovers.G1DListCrossoverUniform)
ga = GSimpleGA.GSimpleGA(genome)
ga.setPopulationSize(100)
ga.setGenerations(100)
ga.evolve(freq_stats=1)
print ga.bestIndividual()
print(NumApp)
print(count)
fo = open(foldermodels+"bestindividual", "a")
eval_func_JustModel(ga.bestIndividual().genomeList)
fo.write(str(ga.bestIndividual()))
logfile.write(str(ga.bestIndividual())+'\n')
fo.close()
logfile.close()
