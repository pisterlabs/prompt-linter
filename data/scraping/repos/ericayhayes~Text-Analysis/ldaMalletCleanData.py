
# coding: utf-8
# Run cyberdh Python evironment
# NOTE: This cell is only for use with Research Desktop. 
# You will get an error if you try to run this chunk of code on your personal device!!
import sys
import os
sys.path.insert(0,"/N/u/cyberdh/Carbonate/dhPyEnviron/lib/python3.6/site-packages")
os.environ["NLTK_DATA"] = "/N/u/cyberdh/Carbonate/dhPyEnviron/nltk_data"

# Import packages
import re
from nltk.corpus import stopwords
import glob
import pandas as pd
from pprint import pprint
from collections import Counter
import matplotlib.pyplot as plt
import pickle
import itertools as it
import zipfile
import tarfile

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.models.phrases import Phrases, Phraser
from gensim.models.wrappers import LdaMallet

# spacy for lemmatization
import spacy

# Import warning
import logging
import warnings


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)
warnings.filterwarnings("ignore",category=DeprecationWarning)
warnings.filterwarnings("ignore",category=UserWarning)
warnings.filterwarnings("ignore",category=FutureWarning)

# Getting your data
# File paths
homePath = os.environ["HOME"]
dataHome = os.path.join(homePath, "Text-Analysis-master", "data")
dataResults = os.path.join(homePath, "Text-Analysis-master", "Output")
malletPath = os.path.join(dataHome,"mallet-2.0.8", "bin", "mallet") # update this path
cleanDataPath = os.path.join(homePath, "Text-Analysis-master", "TopicModeling", "LDA", "cleanedData", "malletModel")
cleanData = os.path.join(cleanDataPath, "ldaDataClean")
cleanDict = os.path.join(cleanDataPath, "ldaDict")
cleanModel = os.path.join(cleanDataPath, "ldaModel")
origData = os.path.join(cleanDataPath, "ldaDataOrig")

# Set needed variables
source = "*"
fileType = ".txt"
docLevel = True
n = 100
nltkStop = True
customStop = True
spacyLem = True
coherence = False
stopLang = 'english'
lemLang = 'en'
encoding = "utf-8"
errors = "ignore"
textColIndex = "text"
stopWords = []
docs = []

# Stopwords
# NLTK Stop words
if nltkStop is True:
    stopWords.extend(stopwords.words(stopLang))
    stopWords.extend(['would', 'said', 'says', 'also'])

# Add own stopword list
if customStop is True:
    stopWordsFilepath = os.path.join(homePath, "Text-Analysis-master", "data", "earlyModernStopword.txt")

    with open(stopWordsFilepath, "r",encoding = encoding) as stopfile:
        stopWordsCustom = [x.strip() for x in stopfile.readlines()]

    stopWords.extend(stopWordsCustom)

# Unzip files
if fileType == ".csv":
    dataRoot = os.path.join(dataHome, "twitter","CSV")
    direct = dataRoot
    allZipFiles = glob.glob(os.path.join(dataRoot,"*.zip"))
    for item in allZipFiles:
        fileName = os.path.splitext(direct)[0]
        zipRef = zipfile.ZipFile(item, "r")
        zipRef.extractall(fileName)
        zipRef.close()
        os.remove(item)
if fileType == ".json":
    dataRoot = os.path.join(dataHome, "twitter","JSON")
    direct = dataRoot
    allZipFiles = glob.glob(os.path.join(dataHome, "twitter","JSON","*.zip"))
    for item in allZipFiles:
        fileName = os.path.splitext(direct)[0]
        zipRef = zipfile.ZipFile(item, "r")
        zipRef.extractall(fileName)
        zipRef.close()
        os.remove(item)
        
if not glob.glob(os.path.join(dataHome, "*.tar.gz")):
    None
else:
    fname = "mallet-2.0.8.tar.gz"
    if fname.endswith(".tar.gz"):
        tar = tarfile.open(os.path.join(dataHome, fname), "r:gz")
        tar.extractall(dataHome)
        tar.close()
        os.remove(os.path.join(dataHome, fname))
        
# Reading in .txt files
if fileType == ".txt":
    paths = glob.glob(os.path.join(dataHome, "shakespeareFolger", source + fileType))
    for path in paths:
        with open(path, "r", encoding = encoding, errors = errors) as file:
             # skip hidden file
            if path.startswith('.'):
                continue
            if docLevel is True:
                docs.append(file.read().strip('\n').splitlines())
            else:
                for line in file:
                    stripLine = line.strip()
                    if len(stripLine) == 0:
                        continue
                    docs.append(stripLine.split())

# Reading in .csv and .json files
if fileType == ".csv":
    allFiles = glob.glob(os.path.join(dataHome, "twitter", "CSV", source + fileType))     
    df = (pd.read_csv(f, engine = "python") for f in allFiles)
    cdf = pd.concat(df, ignore_index=True)
    cdf = pd.DataFrame(cdf, dtype = 'str')
    tweets = cdf[textColIndex].values.tolist()
if fileType == ".json":
    allFiles = glob.glob(os.path.join(dataHome, "twitter", "JSON", source + fileType))     
    df = (pd.read_json(f, encoding = encoding) for f in allFiles)
    cdf = pd.concat(df, ignore_index=True)
    cdf = pd.DataFrame(cdf, dtype = 'str')
    tweets = cdf[textColIndex].values.tolist()

# Data variable
if len(docs) > 0:
    if docLevel is True:
        data = docs
    else:
        data = []
        for i in list(it.zip_longest(*(iter(docs),)*n)):
            data.append(i)
else:
    if len(tweets) > 0:
        data = tweets
        # Remove Urls
        data = [re.sub(r'http\S+', '', sent) for sent in data]
        # Remove new line characters
        data = [re.sub('\s+', ' ', sent) for sent in data]
print(len(data))


# Tokenizing
def sentToWords(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

dataWords = list(sentToWords(data))

print(len(dataWords))

# Find Bigrams and Trigrams
# Variables
minCount = 5
tHold = 100

# Build the bigram and trigram models
bigram = Phrases(dataWords, min_count=minCount, threshold=tHold) # higher threshold fewer phrases.
trigram = Phrases(bigram[dataWords], threshold=tHold)  

# Removes model state from Phrases thereby reducing memory use.
bigramMod = Phraser(bigram)
trigramMod = Phraser(trigram)

# See bigram/trigram example
testNgram = trigramMod[bigramMod[dataWords[0]]]
char = "_"
nGrams = [s for s in testNgram if char in s]
            
pprint(Counter(nGrams))


# Functions
# Define functions for stopwords, bigrams, trigrams and lemmatization
def removeStopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stopWords] for doc in texts]

def makeBigrams(texts):
    return [bigramMod[doc] for doc in texts]

def makeTrigrams(texts):
    return [trigramMod[bigramMod[doc]] for doc in texts]


if spacyLem is True:
    def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        """https://spacy.io/api/annotation"""
        textsOut = []
        lemmaPOS = []
        for sent in texts:
            doc = nlp(" ".join(sent)) 
            textsOut.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
            lemmaPOS.append([token.text and token.lemma_ and token.pos_ for token in doc if token.pos_ in allowed_postags])
        return textsOut
        print(lemmaPOS[:10])

# Remove Stop Words
dataWordsNostops = removeStopwords(dataWords)

# Form Bigrams
dataWordsNgrams = makeBigrams(dataWordsNostops)

if spacyLem is True:
    # Initialize spacy language model, eliminating the parser and ner components
    nlp = spacy.load(lemLang, disable=['parser', 'ner'])
    
    # Do lemmatization tagging only noun, adj, vb, adv
    allowed_postags = ['NOUN', 'ADJ', 'VERB', 'ADV']
    dataLemmatized = lemmatization(dataWordsNgrams, allowed_postags=allowed_postags)
    lemmaPOS = []
    for sent in dataLemmatized:
        lemmaNLP = nlp(" ".join(sent))
        for token in lemmaNLP:
            lemmaPOS.append([token.text, token.lemma_, token.pos_])
    print(lemmaPOS[:10])
    

    # Find ngrams and count number of times they occur
    dataNgrams = [s for s in dataLemmatized[0] if char in s]
    
else:
    dataNgrams = [s for s in dataWordsNgrams[0] if char in s]
print(Counter(dataNgrams))


# reate the Dictionary and Corpus needed for Topic Modeling
if spacyLem is True:
    # Create Dictionary
    id2word = corpora.Dictionary(dataLemmatized)

    # Create Corpus
    texts = dataLemmatized
else:
    # Create Dictionary
    id2word = corpora.Dictionary(dataWordsNgrams)

    # Create Corpus
    texts = dataWordsNgrams
    
# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# Test dictionary
id2word[10]

# Human readable format of corpus (term-frequency)
hReadable = [[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]]
for i in hReadable:
    print(i[:10])


# USING MALLET
# Variables
nTopics = 20
workers = 1
nIter = 1000
seed = 42

ldamallet = LdaMallet(malletPath, corpus=corpus, num_topics=nTopics, id2word=id2word, workers = workers, prefix = cleanDataPath,iterations = nIter, random_seed = seed)

# Show Topics
pprint(ldamallet.show_topics(formatted=False)[:3])
if coherence is True:
    if spacyLem is True:
        # Compute Coherence Score
        coherenceModelLdamallet = CoherenceModel(model=ldamallet, texts=dataLemmatized, dictionary=id2word, coherence='c_v')
        coherenceLdamallet = coherenceModelLdamallet.get_coherence()
    else:
        # Compute Coherence Score
        coherenceModelLdamallet = CoherenceModel(model=ldamallet, texts=dataWordsNgrams, dictionary=id2word, coherence='c_v')
        coherenceLdamallet = coherenceModelLdamallet.get_coherence()
    
    print('\nCoherence Score: ', coherenceLdamallet)
else:
    None

# FIND OPTIMAL NUMBER OF TOPICS
if coherence is True:
    def computeCoherenceValues(dictionary, corpus, texts, limit, start=20, step=10):
        """
        Compute c_v coherence for various number of topics

        Parameters:
        ----------
        dictionary : Gensim dictionary
        corpus : Gensim corpus
        texts : List of input texts
        limit : Max num of topics

        Returns:
        -------
        modelList : List of LDA topic models
        coherenceValues : Coherence values corresponding to the LDA model with respective number of topics
        """
        coherenceValues = []
        modelList = []
        for numTopics in range(start, limit, step):
            model = gensim.models.wrappers.LdaMallet(malletPath, corpus=corpus, num_topics=numTopics, id2word=id2word)
            modelList.append(model)
            coherenceModel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
            coherenceValues.append(coherenceModel.get_coherence())

        return modelList, coherenceValues
else:
    None

# Can take a long time to run.
if coherence is True:
    nt = 20
    lmt = 81
    steps = 10
    if spacyLem is True:
        modelList, coherenceValues = computeCoherenceValues(dictionary=id2word, corpus=corpus, texts=dataLemmatized, start=nt, limit=lmt, step=steps)
    else:
        modelList, coherenceValues = computeCoherenceValues(dictionary=id2word, corpus=corpus, texts=dataWordsNgrams, start=nt, limit=lmt, step=steps)
else:
    None

# Plot line graph showing coherence values
if coherence is True:
    # Show graph
    limit=lmt; start=nt; step=steps;
    x = range(start, limit, step)
    plt.plot(x, coherenceValues)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherenceValues"), loc='best')
    plt.show()
else:
    None

# list the coherence score for each number of topics we have selected. 
if coherence is True:
    # Print the coherence scores
    for m, cv in zip(x, coherenceValues):
        print("Num Topics =", m, " has Coherence Value of", round(cv, 4))
else:
    None

# Choose optimal number of topics
if coherence is True:
    modelNum = 0
    # Select the model and print the topics
    optimalModel = modelList[modelNum]
    modelTopics = optimalModel.show_topics(formatted=False)
    pprint(optimalModel.print_topics(num_words=10)[:3])
else:
    optimalModel = ldamallet
    modelTopics = optimalModel.show_topics(formatted=False)
    pprint(optimalModel.print_topics(num_words=10)[:3])


# Save models, dictionary, and data
optimalModel.save(cleanModel)
id2word.save(cleanDict)
with open(origData, "wb") as sd:
    pickle.dump(data, sd)
with open(cleanData, "wb") as phrases:
    pickle.dump(texts, phrases)

# Ackowledgements: This algorithm was adapted from the blog "Machine Learning Plus". Reference: Machine Learning Plus. Topic Modeling with Gensim (Python). Retrieved from https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/ on November 5, 2018.
