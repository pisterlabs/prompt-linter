# ---
# jupyter:
#   jupytext:
#     comment_magics: false
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.1.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# ANLY 580 - NLP for Data Analytics
# Fall Semester 2019

# %% [markdown]
# ### Topic Modleing - Latent Dirichlet Allocation (LDA) and Gensim

# %%
import os
import pandas as pd
import numpy as np

import gensim
from gensim import corpora, models, similarities
from gensim.models import CoherenceModel, HdpModel

from nltk.corpus import stopwords
import string
import re
import pprint

from collections import OrderedDict

import seaborn as sns

import pyLDAvis.gensim

import matplotlib.pyplot as plt
%matplotlib inline

import warnings
warnings.filterwarnings("ignore")

# %%
# Modify these parmeters for the local directory structre where the data os located. 

data_file_path = "data"
temp_file_path = "backup"

# %% [markdown]
# #### Use the train and test data from Semval 2016 Task BD, which includes human annotated topics

# %%
# As shown below this example uses the Semval Gold training and test sets from subtask BD 
# which include human annotated tags for use in comparing topic modeling
# interpreation of the content.

# Load the data into a pandas dataframe. 

# Set to True to include additional data from the "test" set. 
INCLUDE_TRAIN = True  # Include the train set by default
INCLUDE_TEST = False  # Set to True to also include the test set, results in longer run times 

# Read the train data into a pandas dataframe 
datafile = os.path.join(data_file_path, "2017_English_final/GOLD/Subtasks_BD/twitter-2016train-BD.txt")
tweets1 = pd.read_csv(datafile, 
                     encoding = 'utf-8', 
                     sep = '\t', 
                     header = None,
                     index_col = False,
                     names = ['msgid', 'topic', 'sentiment', 'Tweet'], 
                     dtype = {'msgid':str, 'topic':str, 'sentiment':str, 'Tweet':str})

# Read the test data into a pandas dataframe
datafile = os.path.join(data_file_path, "2017_English_final/GOLD/Subtasks_BD/twitter-2016test-BD.txt")
tweets2 = pd.read_csv(datafile, 
                     encoding = 'utf-8', 
                     sep = '\t', 
                     header = None,
                     index_col = False,
                     names = ['msgid', 'topic', 'sentiment', 'Tweet'], 
                     dtype = {'msgid':str, 'topic':str, 'sentiment':str, 'Tweet':str})


if INCLUDE_TRAIN and INCLUDE_TEST:
    tweets = pd.concat([tweets1, tweets2], ignore_index=True)
elif INCLUDE_TRAIN:
    tweets = tweets1
else:
    tweets = tweets2


# %%
# How many human annotated topics are in the data?

human_topics = list(set(tweets['topic'].tolist()))
print("Number of human topics in the data: {}".format(len(human_topics)))

# %%
# Let's look at the shape of the data, how many tweets are in the data set?
print("Shape of the data: {}".format(tweets.shape))

# And take a look at the first 10 rows
tweets.head(10)

# %%
# And take a look at the last 10 rows
tweets.tail(10)

# %%
# Take a look at the distributions of human annotated/tagged topics in the data via a barplot

sns.set(rc={'figure.figsize':(12.7,9.27)})
by_topic = sns.countplot(x='topic', data=tweets)

for item in by_topic.get_xticklabels():
    item.set_rotation(90)

# %%
# Take a look at the distributions of human annotated/tagged sentiment by topic via a barplot

human_sentiment = list(set(tweets['sentiment'].tolist()))
df_sentiment = tweets.groupby(['topic', 'sentiment'])['topic'].count().unstack('sentiment')
topic_mixture = df_sentiment[human_sentiment].plot(kind='bar', stacked=True, legend = True)

# %%
# What does the human anotated set of topics look like
human_topics = list(set(tweets['topic'].tolist()))
print(human_topics)

# %%
# Prepare the corpus for analysis and checking first 10 entries

corpus = []

for i in range(len(tweets['Tweet'])):
    tweet = tweets['Tweet'][i]
    
    # For topic modeling we will remove the url's for consdieration as terms in our topics
    #tweet = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', "", tweet)
    tweet = re.sub(r'http\S+', '', tweet)
    
    # Uncomment the following to remove hashtags and mentions
    #tweet = re.sub(r'#\w+ ?', '', tweet)
    #tweet = re.sub(r'@\w+ ?', '', tweet)
    
    # OR, uncomment the following to remove the character tags for mentions and hashtags
    #tweet = tweet.replace("@", "").replace("#", "")
    
    tweet = tweet.replace("&amp;", " ").replace("&gt;", "").replace("&lt;", "")
    tweet = tweet.replace("(", "").replace(")", "").replace(".", "").replace("?", "").replace("!", "").replace(",", "")
    tweet = tweet.replace("/", " ").replace("=", "").replace('\"', "").replace('*', '').replace(';', "")
    tweet = tweet.replace(':', '').replace('"', '')
    tweet = re.sub(r'\$[0-9]+', '', tweet)
    tweet = re.sub(r'[0-9]+GB', '', tweet)
    tweet = re.sub(r'[0-9]+', '', tweet)
    tweet = re.sub(r'--+', ' ', tweet)
    
    corpus.append(tweet)

# Dump out the first 10 tweets to see what the parsing has done
corpus[0:10]

# %%
# If our temp folder is not present, create it. 

TEMP_FOLDER = temp_file_path
if os.path.exists(TEMP_FOLDER) == False:
    os.mkdir(TEMP_FOLDER)

# %% [markdown]
# #### gensim LDA does not use the words directly when determining topics but instead uses ids as representations for the words. The mapping between ids and words is stored in a python dictionary. 

# %%
# Now that we have done some prep of the corpus we can perform some additonal word level processing to remove 
# extraneous tokens and common stopwords that do not contribute our discovery of topics in the corpus.

# Define our stoplist for removing common words and tokenizing
list1 = ['RT','rt', '&amp;', 'im', 'b4', 'yr', 'nd', 'rd', 'oh', "can't", "he's", "i'll",
         "i'm", 'ta', "'s", "c'mon", 'th', 'st', "that's", "they're", "i've", 'am', 'pm']
stoplist = stopwords.words('english') + list(string.punctuation) + list1
#print(stoplist)

# Remove tokens in the text that match our stoplist tokens and lower case all tokens
texts = [[word for word in str(document).lower().split() if word not in stoplist] for document in corpus]

# Create and save the dictionary, in case we want to reload it into another notebook
dictionary = corpora.Dictionary(texts)
dictionary.save(os.path.join(TEMP_FOLDER, 'semval.dict'))  # store the dictionary, for future reference

# %%
# What is the averge length of the documents in the corpus?
text_length = []
for t in texts:
    text_length.append(len(t))
    
tweets['doc_length'] = pd.Series(text_length)

avg_doc_length = tweets['doc_length'].mean() 
median_doc_length = tweets['doc_length'].median()
min_doc_length = tweets['doc_length'].min()
max_doc_length = tweets['doc_length'].max()

print("Average tweet length: {}".format(avg_doc_length))
print("Median tweet length: {}".format(median_doc_length))
print("Minimum tweet length: {}".format(min_doc_length))
print("Maximum tweet length: {}".format(max_doc_length))


# %%
# Found some anomalies in the input training data set that resulted in very long tweets.
# Code below is a chedk that I corrected all the anomalies

tweets_filtered = tweets[tweets['doc_length'] > 40]
tweets_filtered.head()

outliers = tweets_filtered['Tweet'].tolist()
print(outliers)

# %%
# Dump out the dictionay to examine list of resulting tokens
print(dictionary.token2id)

# %% [markdown]
# #### gensim LDA stores all the text for processing into a corpus object. All text is filtered through the previously constructed dictionary

# %%
corpus = [dictionary.doc2bow(text) for text in texts]
corpora.MmCorpus.serialize(os.path.join(TEMP_FOLDER, 'semval.mm'), corpus)  # store to disk, for later use

# %%
tfidf = models.TfidfModel(corpus) # step 1 -- initialize a model
corpus_tfidf = tfidf[corpus]      # step 2 -- use the model to transform vectors

# %% [markdown]
# ##### Reference for lda parameters: https://radimrehurek.com/gensim/models/ldamodel.html

# %%
# Use the number of human annotated topics by commenting out the second total_topics assignment 
total_topics = len(human_topics)
total_topics = 20

# Experiment with the alpha assingment to investigate alpha parameter settings on topic assignments
lda_alpha = 'auto' #learns asymmetic prior from the corpus
lda_alpha = 'symmetric'
#lda_alpha = 'asymmetric'  # sets alpha = 1 / number_of_topics
#lda_alpha = np.full((total_topics), (0.05 * avg_doc_length) / total_topics)  # from NIH paper

# %%
#
lda = models.LdaModel(corpus, id2word = dictionary, num_topics = total_topics, iterations = 1000, alpha=lda_alpha)
corpus_lda = lda[corpus] # Use the bow corpus

# %%
#Show first n=80 important words in the topics:
lda.show_topics(total_topics, 10)

# %%
# Load the topic - term data into an python dictionary
data_lda = {i: OrderedDict(lda.show_topic(i,20)) for i in range(total_topics)}
data_lda

# %%
# Use the ordered dictionary to load the data into a dataframe
df_lda = pd.DataFrame(data_lda)
df_lda = df_lda.fillna(0).T
print(df_lda.shape)

# %%
# A dataframe view of some of terms across topics

df_lda.head(20)

# %%
#pyLDAvis.enable_notebook()
#panel = pyLDAvis.gensim.prepare(lda, corpus, dictionary, mds='tsne')
#panel

# %% [markdown]
# #### We have some idea of the topic distribution from the preceding cell but no real idea how lda matched the human annotated topics. 

# %%
parsed_tweets = tweets.filter(['msgid', 'topic', 'sentiment'], axis =1)
se = pd.Series(texts)
parsed_tweets['Tweet'] = se

# %%
parsed_tweets.to_csv(os.path.join(data_file_path, "parsed_tweets.csv"), sep="\t")
parsed_tweets.head(10)

# %%
# A check of the data types included in parsed_tweets

#parsed_tweets.dtypes

# %%
# Run the original documents back thru the model to infer the distribution of topics 
# according to the lda model

topics = []
probs = []
max_to_show = 20

for k, i in enumerate(range(len(parsed_tweets['Tweet']))):
    bow = dictionary.doc2bow(parsed_tweets['Tweet'][i])
    doc_topics = lda.get_document_topics(bow, minimum_probability = 0.01)
    topics_sorted = sorted(doc_topics, key = lambda x: x[1], reverse = True)
    topics.append(topics_sorted[0][0])
    probs.append("{}".format(topics_sorted[0][1]))
    
    # Dump out the topic and probability assignments for the first 20 documents
    if k < max_to_show:
        print("Document {}: {}".format(k, topics_sorted))

parsed_tweets['LDAtopic'] = pd.Series(topics)
parsed_tweets['LDAprob'] = pd.Series(probs)

# %%
# Resort the dataframe according to the human annotated topic and lda topic
parsed_tweets.sort_values(['topic', 'LDAtopic'], ascending=[True, True], inplace=True)
parsed_tweets.head(20)

# %%
# Take a look at the distributions of human annotated topics in the data via a barplot

sns.set(rc={'figure.figsize':(12.7,9.27)})
by_topic = sns.countplot(x='LDAtopic', data=parsed_tweets)

for item in by_topic.get_xticklabels():
    item.set_rotation(90)

# %%
# Resort the dataframe according to the the lda assigned topic and the human annotated topic

parsed_tweets.sort_values(['LDAtopic', 'topic'], ascending=[True, True], inplace=True)
parsed_tweets.head(20)

# %%
# Resort the dataframe according to the the lda assigned topic and the assocoiated probability
parsed_tweets.sort_values(['LDAtopic', 'LDAprob'], ascending=[True, False], inplace=True)
parsed_tweets.head(20)

# %%
# What do the topic distrubtions look like relative to the original human annotated/tagged topics

df2 = parsed_tweets.groupby(['LDAtopic', 'topic'])['LDAtopic'].count().unstack('topic')
topic_mixture = df2[human_topics].plot(kind='bar', stacked=True, legend = False)


# %%
# What do the topic distrubtions look like relative to the original human annotated/tagged sentiment

human_sentiment = list(set(parsed_tweets['sentiment'].tolist()))
df2 = parsed_tweets.groupby(['LDAtopic', 'sentiment'])['LDAtopic'].count().unstack('sentiment')
topic_mixture = df2[human_sentiment].plot(kind='bar', stacked=True, legend = True)

# %%
# A major question in using LDA for topic modeling is what is is the proper set of
# hyperparmeters to generate the optimal set of topics for the coprus of documents
# under examination. Gensim includes methods for computing the Perplexity and Topic 
# Coherence of a corpus. One appraoch to is to sample an LDA model for a range of 
# for perplexity and topic coherence and select the appropriate number of topics
# from a point of minimum perplexity and maximium topic coherence.

corpus = [dictionary.doc2bow(text) for text in texts]
perplexity_lda = []
coherence_lda = []
topic_count_lda = []

for num_topics in range(15, 70, 5):
    
    print("Computing the lda model using {} topics".format(num_topics))
    
    topic_lda = models.LdaModel(corpus,
                                id2word = dictionary,
                                num_topics = total_topics,
                                iterations = 1000,
                                alpha = lda_alpha)
    corpus_lda = topic_lda[corpus] # Use the bow corpus
    
    topic_count_lda.append(num_topics)
    
    perplexity_lda.append(topic_lda.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

    # Compute Coherence Score
    cm = CoherenceModel(model=topic_lda, corpus=corpus, dictionary=dictionary, coherence='u_mass')
    coherence_lda.append(cm.get_coherence())
    

#cm = CoherenceModel(model=lda, corpus=corpus, dictionay=dictionary, coherence='c_v')
#coherence_lda = cm.get_coherence()
#print('\nCoherence Score (c_v): ', coherence_lda)

# %%
# Pull the resulting data into a pandas dataframe
topics_lda = pd.DataFrame({'perplexity': perplexity_lda,
                           'coherence': coherence_lda},
                         index = topic_count_lda)

topics_lda.head(10)

# %%
lines = topics_lda.plot.line(subplots = True)

# %% [markdown]
# ##### Gensim also includes Hierarchical Dirichlet process (HDP). HDP is a powerful mixed-membership model for 
# the unsupervised analysis of grouped data. Unlike its finite counterpart, latent Dirichlet allocation, 
# the HDP topic model infers the number of topics from the data. Here we have used Online HDP, 
# which provides the speed of online variational Bayes with the modeling flexibility of the HDP.
#
# See https://radimrehurek.com/gensim/models/hdpmodel.html

# %%
# Create a HDP model - default for hdp is 150
hdpmodel = HdpModel(corpus=corpus, id2word=dictionary)

# %%
hdptopics = hdpmodel.show_topics(num_topics = 20, formatted=True)
hdptopics

# %%
hdp_topics = hdpmodel.get_topics()
hdp_topics.shape

# %%
hdpmodel.hdp_to_lda()
