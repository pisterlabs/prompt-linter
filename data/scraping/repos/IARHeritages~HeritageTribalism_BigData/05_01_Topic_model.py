### Topic model ###
## This code creates the LDA topic model for the unique tweets in the dataset, for topic numbers n = 2:28,
## it calculates the coherence scores for the topic and creates and intertopic-distance visualisation for the model with the highest coherence score, with the LDAvis.
## Then it also assignes the dominant topic to the tweets


#Import relevant libraries:
import pymongo
import time
import gensim
import os
import csv
import re
import operator
import warnings
import numpy as np
from gensim.models import CoherenceModel, LdaModel, LsiModel, HdpModel
from gensim.models.wrappers import LdaMallet
from gensim.corpora import Dictionary
from gensim.parsing.preprocessing import strip_punctuation
from pprint import pprint
from gensim.corpora.mmcorpus import MmCorpus
from gensim.models import ldamulticore
from gensim import models
from gensim.utils import lemmatize
from nltk.corpus import stopwords
import pyLDAvis.gensim
import gc
import logging
from gensim.utils import lemmatize
from nltk.corpus import stopwords
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.models import Phrases
from gensim.corpora import MmCorpus
from gensim.test.utils import get_tmpfile 
from collections import OrderedDict

# Set up logging to console to monitor the progress of the topic model:
console = logging.StreamHandler()
console.setLevel(logging.INFO)
# set a format which is simpler for console use
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
console.setFormatter(formatter)
# add the handler to the root logger
logging.getLogger('').addHandler(console)
logger = logging.getLogger(__name__)

### Set up workspace

# Set working directory
os.chdir('~path')

# Define the paths to inputs:
path2tweets='data/tweets.csv'

# Define the paths to outputs
path2corpus='outputs/topic_models/corpus'
path2dictionary='outputs/topic_models/dictionary'
path2model= 'outputs/topic_models/models_'
path2coherence = 'outputs/03_01_01_coherenceScores.csv' # Path to model coherence scores
path2html = 'outputs/03_01_02_topic_model.html' # Path to the best model visualisation in html

# Define language to use for the model, and the threshold for the number of topics
language='english'
max_topics=31

# Import the texts of the tweets:
with open(path2tweets,encoding="utf8") as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    tweets = []
    for row in readCSV:
        tweet = row[1]
        tweets.append(tweet)
        
# Delete the column name
del tweets[0]

### Texts processing

# Get rid of links usernames, and indication that the tweet is a retweet from the text:
i=0
j=len(tweets)
while(i<j):
    tweets[i] = re.sub('http\S+', '', tweets[i])
    tweets[i] = re.sub('@\S+', '', tweets[i])
    tweets[i] = re.sub('RT|cc', '', tweets[i])
    i=i+1

#Change tweets into token lists:
i=0
j=len(tweets)
while(i<j):
    print(i)
    tweets[i] = gensim.utils.simple_preprocess(tweets[i], deacc=True, min_len=3)
    i=i+1
    
# Get only unique texts:
texts= []
for tweet in tweets:
    if tweet not in texts:
      texts.append(tweet)
        
# Import and define stpowords:
nltk.download('stopwords')
stops = set(stopwords.words('english'))
#Also add new stopwors, which are not infomrative, because they are in every tweet
new_stops = set(["man","cheddar","cheddarman"])
                 
## Get rid of english stopwords and user defined stopwords:
texts = [[word for word in text if word not in stops] for text in texts]
texts = [[word for word in text if word not in new_stops] for text in texts]

#Lemmatize all the words in the document:
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
texts= [[lemmatizer.lemmatize(token) for token in text] for text in texts]

# Create bigrams and trigrams:

# Add bigrams and trigrams to docs (only ones that appear 20 times or more).
bigram = Phrases(texts, min_count=20)
for idx in range(len(texts)):
    for token in bigram[texts[idx]]:
        if '_' in token:
            # Token is a bigram, add to document.
            texts[idx].append(token)

# Make dictionary and the corpus
train_texts = texts 
dictionary = Dictionary(train_texts)
corpus = [dictionary.doc2bow(text) for text in train_texts]
    
### Save corpus and dictionary:  
MmCorpus.serialize(path2corpus, corpus)
mm = MmCorpus(path2corpus)
dictionary.save_as_text(path2dictionary)
dictionary = Dictionary.load_from_text(path2dictionary)

# Set up the list to hold coherence values for each topic:
c_v = []
# Loop over to create models with 2 to 30 topics, and caluclate coherence scores for it:
for num_topics in range(2, max_topics):
    print(num_topics)
    lm = models.LdaMulticore(corpus=mm, num_topics=num_topics,     id2word=dictionary,chunksize=9000,passes=100,eval_every=1,iterations=500,workers=4) # Create a model for num_topics topics
    print("Calculating coherence score...")
    cm = CoherenceModel(model=lm, texts=train_texts, dictionary=dictionary, coherence='c_v') # Calculate the coherence score for the topics
    print("Saving model...")
    lm.save(path2model+str(num_topics)) # Save the model
    lm.clear() # Clear the data from the model
    del lm # Delete the model
    gc.collect() # Clears data from the workspace to free up memory
    c_v.append(cm.get_coherence()) # Append the coherence score to the list


# Save the coherence scores to the file:    
with open(path2coherence, 'a') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerow(["n_topics","coherence_score"])
    i=2
    for score in c_v:
        print(i)
        writer.writerow([i,score])
        i=i+1
        
#Get the best topic model and construct the visualisation

n=c_v.index(max(c_v))+2 # Get the number of topics with the highest coherence score
lm = LdaModel.load(path2model+str(n)) # Load the number of topics with the highest coherence score into the workspace
tm = pyLDAvis.gensim.prepare(lm, mm, dictionary) # Prepare the visualisation
pyLDAvis.save_html(tm, path2html+str(n)+'.html') # Save the visualisation


### Assign topics to tweets, using their ids:
# Reorder topic:

# Change the topics order to be consistent with the to be consistent with the pyLDAvis topic model (ordered from the most
# frequent one to the least frequent one) and assign dominant topic to each tweet:

# Get the topic order
to=tm.topic_order

# set up writing to a file
with open(path2tweets, 'w') as f: 
    w = csv.DictWriter(f, field_names)
    w.writeheader()
    # Loop over all the tweets in the corpus, and assign topic to each:
    for i in range(0,len(corpus)+1):
        topics = lm.get_document_topics(corpus[i]) # Get topic probabilities for the document
        topics=list(topics)    
        topics=[list(topic) for topic in topics] # Reformat topics probabilities for the analysis
        # reorder topics according to pyLDAvis numbering
        topics=[["Topic "+str(to.index(topic[0]+1)+1),topic[1]] for topic in topics] 
        topics = sorted(topics)
        topics = dict(topics)
        # Get dominant value and dominant topic (the highest probability and the topic with the highest probability) for the tweet.
        topics['dominant_topic'] = max(topics,key=topics.get)
        topics['dominant_value'] = topics[topics['dominant_topic']]
        topics["_id"] = ids[i]
        topics["text"] = texts[i]
        # Write dominant value and dominant topic, together with the values of all topics to the file:
        w.writerow(topics)       