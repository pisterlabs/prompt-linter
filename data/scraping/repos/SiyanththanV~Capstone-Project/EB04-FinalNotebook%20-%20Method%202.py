#!/usr/bin/env python
# coding: utf-8

# # **Capstone EB04**

# The following version of the code is for finding like minded user communities by getting higher level topics during the lda phase before performing clustering. This is the second of two different methods that were attempted.

# ## Imports

# In[1]:


import os 
import csv
import json
import datetime
import ast
import re
import numpy as np
import pandas as pd
from pprint import pprint

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim import models

# spacy for lemmatization
import spacy
import json
import warnings
import networkx as nx

warnings.filterwarnings("ignore",category=DeprecationWarning)

from langdetect import detect
from langdetect import DetectorFactory
DetectorFactory.seed = 0
import numpy as np
import pandas as pd
from pprint import pprint
import pickle


# Plotting tools
import pyLDAvis
import pyLDAvis.gensim 
import matplotlib.pyplot as plt
import sklearn
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

import nltk
nltk.download('stopwords')
nltk.download('words')

# NLTK Stop words
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

from nltk.corpus import words
eng_words = words.words('en')


# ## Load Preprocessed User Tweet Data

# In[2]:


#loads csv from stored location
df = pd.read_csv('../csvfiles/tweetsOnUserOnConcepts.csv', lineterminator='\n', low_memory=False)
df.head()


# In[3]:


#gets all concept text for each tweet and stores in list
tweetConcept = df.ConceptText.values.tolist()
userIds = df.userid.values.tolist()


# In[4]:


#stores a list of each tweet and the words the tweets contain
tempData = []

for sent in tweetConcept:
    x = []
    for word in sent.split(" "):
        x.append(word)
    tempData.append(x)

data_final = tempData


# ## LDA Analysis

# In[5]:


#setting up corpus for lda
id2word1 = corpora.Dictionary(data_final)
texts = data_final
corpus1 = [id2word1.doc2bow(text) for text in texts]


# ### Run LDA
# Multicore allows for multiple cores to be working on LDA simultaneously
# - Check Number of workers
# - Check Number of topics set<br>

# In[6]:


#uncomment line below to try lda with different values
topicNum = 47
#lda_model = gensim.models.ldamulticore.LdaMulticore(corpus=corpus1,id2word=id2word1,num_topics=topicNum, passes=10, workers=7)
#lda_model.save('../LdaSaves/topics47mar10p9/lda.model_mar10_t47')

#preloading an saved lda run to save time as lda takes long time to run
lda_model1 =  models.LdaModel.load('../LDASaves/ldamar25/lda.model_mar25_t47')
pprint(lda_model1.print_topics())


# In[30]:


#creating a dictionary where each key is the user and the value is a list of all topicsNums representing each tweet
dictConcept = {}

for i in range(len(userIds)):
    if userIds[i] not in dictConcept:
        dictConcept[userIds[i]] = []

#getting the top 3 topics for each user tweet and appending to the user dictionary
for i, row in enumerate (lda_model1[corpus1]):
    sortedValue = sorted(row, key=lambda x:x[1], reverse=True)
    userid =int(df.iloc[[i]].userid)
    for z in sortedValue[:3]:
        dictConcept[userid].append(str(z[0]))


# ### Running 2nd Round of LDA To find Higher Level Topics

# In[11]:


import ast
#using previously saved list to save time
# topicsPerTweets=[]
# with open('topicsPerTweets.txt', 'r') as f:
#     for line in f:
#         topicsPerTweets.append(ast.literal_eval(line))
        
topicsPerTweets = list(dictConcept.values())
print(topicsPerTweets[:2])

id2word2 = corpora.Dictionary(topicsPerTweets)
corpus2 = [id2word2.doc2bow(text) for text in topicsPerTweets]

#saving the list so the above block does not need to be rerun since it takes a while
# with open('topicsPerTweets.txt', 'w') as f:
#     for item in topicsPerTweets:
#         f.write("%s\n" % item)


# In[12]:


topicNum2 = 10
today = datetime.datetime.now()
#uncomment the line below to run with own custom topics numbers or workers
lda_model2 = gensim.models.ldamulticore.LdaMulticore(corpus=corpus2, id2word=id2word2, num_topics=topicNum2, passes=10, workers=3)
lda_model2.save('../LDASaves/HigherOrderModels/LDA' + today.strftime("%M%d") + str(topicNum2))

lda_model2 =  models.LdaModel.load('../LDASaves/HigherOrderModels/LDA' + today.strftime("%M%d") + str(topicNum2))
doc_lda2 = lda_model2[corpus2]
pprint(lda_model2.print_topics())


# #### Compute Perplexity and Coherence

# In[14]:


print('\nPerplexity: ', lda_model2.log_perplexity(corpus2))  # a measure of how good the model is. lower the better.
coherence_model_lda = CoherenceModel(model=lda_model2, texts=topicsPerTweets, dictionary=id2word2, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)


# #### Finding Topic Distribution

# In[48]:


def format_topics_sentences(ldamodel=lda_model2, corpus=corpus2, texts=topicsPerTweets):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)


df_topic_sents_keywords = format_topics_sentences(ldamodel=lda_model2, corpus=corpus2, texts=topicsPerTweets)

# Format
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

# Show
df_dominant_topic.head(10)


# In[51]:


# Number of Documents for Each Topic
topic_counts = df_topic_sents_keywords['Dominant_Topic'].value_counts()

# Percentage of Documents for Each Topic
topic_contribution = round(topic_counts/topic_counts.sum(), 4)

# Topic Number and Keywords
topic_num_keywords = df_topic_sents_keywords[['Dominant_Topic', 'Topic_Keywords']]

# Concatenate Column wise
df_dominant_topics = pd.concat([topic_num_keywords, topic_counts, topic_contribution], axis=1)

# Change Column names
df_dominant_topics.columns = ['Dominant_Topic', 'Topic_Keywords', 'Num_Documents', 'Perc_Documents']

# Show
df_dominant_topics[0:10]


# #### Creating User Vectors of length K where K is number of topics

# In[15]:


UserVectors = []

#for each users shows percent contribution for that topic
print(lda_model2[corpus2][1][1])

for row in lda_model2[corpus2]:
    temp = [0]*topicNum2
    for val in row:
        #val is a tuple in form (topicNum, percentContributionOfTopicToUser)
        temp[val[0]] = val[1]
    UserVectors.append(temp)
    
print("Shows a sample userVector")    
print(UserVectors[1])


# ## Load Preprocessed Gold Standard News Articles

# In[16]:


dfGoldStandard = pd.read_csv('../csvfiles/GoldStandard.csv',  lineterminator='\n', low_memory=False)
dfGoldStandard.head()


# In[17]:


newsUserId = dfGoldStandard.userid.values.tolist()
newsUrl = dfGoldStandard.url.values.tolist()
newsId = dfGoldStandard.NewsId.values.tolist()

#dictionary of users who posted a newsArticle
newsId2UserId = {}

for i in range(len(newsId)):
    if newsId[i] not in newsId2UserId:
        newsId2UserId[newsId[i]] = []
    newsId2UserId[newsId[i]].append(newsUserId[i])


# In[18]:


# loading another dataframe with goldstandard but only keeping unique newsids
dfUniqueNewsId = pd.read_csv('../csvfiles/GoldStandard.csv',  lineterminator='\n', low_memory=False)
dfUniqueNewsId.drop_duplicates(subset='NewsId', inplace = True)
newsArticles = dfUniqueNewsId.NewsConceptText.values.tolist()


# ### Getting 1st Round LDA topics for news articles

# In[20]:


#storing words in news articles in a list
newsArticlesForCorpus = [x.split(' ') for x in newsArticles]
#creating a corpus
newsId2word = corpora.Dictionary(newsArticlesForCorpus)
NewsArticlesCorpus = [newsId2word.doc2bow(text) for text in newsArticlesForCorpus]

#using the previous lda_model for first lda run with the news corpus created to get a percent contribution for each topic for each news article
TopicDistributionOnNewsArticles = lda_model1[NewsArticlesCorpus]


# In[22]:


# finding the top 3 topics for each news article
topicsPerNewsArticleHighLevel = []
for x in (TopicDistributionOnNewsArticles):
    sortedValue = sorted(x, key=lambda x:x[1], reverse=True)
    temp = []
    for z in sortedValue[:3]:
        temp.append(str(z[0]))
    topicsPerNewsArticleHighLevel.append(temp)

print(topicsPerNewsArticleHighLevel[:5])


# ### Getting 2nd Round LDA topics (High Level Topics) for news articles

# In[23]:


#setting up corpus for 2nd lda run on news articles
newsId2word2 = corpora.Dictionary(topicsPerNewsArticleHighLevel)
NewsArticlesCorpus2 = [newsId2word2.doc2bow(text) for text in topicsPerNewsArticleHighLevel]
TopicDistributionOnNewsArticles = lda_model2[NewsArticlesCorpus2]


# #### Creating User Vectors of length K where K is number of topics

# In[24]:


ArticleVector = []

for row in TopicDistributionOnNewsArticles:
    temp = [0]*topicNum2
    for val in row:
        #val is a tuple in form (topicNum, percentContributionOfTopicToUser)
        temp[val[0]] = val[1]
    ArticleVector.append(temp)
    
print("Displaying sample article vector")
print(ArticleVector[1])


# ## Clustering

# ### Storing and preloading Kmeans results

# In[25]:


#different cluster sizes to try out analysis for
numClusters=[5, 10, 15, 20, 25, 30]
today = datetime.datetime.now()

#saving kmeans results for the differnt cluster sizes
for x in range(len(numClusters)):
    userVectorsFit = np.array(UserVectors)
    #performing kmeans on the userVector to cluster users into communities
    kmeans = KMeans(n_clusters=numClusters[x], random_state=0).fit(userVectorsFit)
    
    kMeansfilename = 'LDAM2-kMeans'+ today.strftime("%M%d") + 'CSize' + str(numClusters[x])
    pickle.dump(kmeans, open("../kmeansFiles/" + kMeansfilename,'wb'))


# In[26]:


#change this number to a number from the [5, 10, 15, 20, 25, 30] to preload a different file
chosenNumberOfCluster = 30

#loading existing kmeans model
kMeansfilename = 'LDAM2-kMeans' + today.strftime("%M%d") + 'CSize' + str(chosenNumberOfCluster)
print('Chosen File: \''+kMeansfilename+'\'')

loadedKmeansModel = pickle.load(open("../kmeansFiles/" + kMeansfilename, 'rb'))


# ### Number of Users in each Cluster
# 

# In[27]:


#creating a list to show how many users are in each cluster
userClusters = [0]*chosenNumberOfCluster
for i in loadedKmeansModel.labels_:
    userClusters[i] += 1

print(userClusters)


# ### User Indexes in each cluster, organized as an array
# 

# In[31]:


UserIndexInCluster=[]
idsDict = list(dictConcept.keys())

for x in range(chosenNumberOfCluster):
    UserIndexInCluster.append([])
    
for index, val in enumerate(loadedKmeansModel.labels_):
    UserIndexInCluster[val].append(index)


# ### User ***IDs*** in each cluster, organized as an array
# 

# In[32]:


idsCluster = []
for x in range(chosenNumberOfCluster):
    idsCluster.append([])
    
for index, val in enumerate(loadedKmeansModel.labels_):
    idsCluster[val].append(idsDict[index])   


# ### Find Topic Distribution Per Cluster

# In[33]:


topicDistributionPerCluster=[]
for x in range(chosenNumberOfCluster):
    topicDistributionPerCluster.append([])
    
for i,cluster in enumerate(UserIndexInCluster):
    for userIndex in cluster:
        topicDistributionPerCluster[i].append(UserVectors[userIndex])


# ### Find Average Topic distribution per Cluster
# 

# In[34]:


averageDistributionPerCluster = []
for x in topicDistributionPerCluster:
    y = np.array(x)
    listOfAverageValues = np.mean(y,axis=0)
    averageDistributionPerCluster.append(listOfAverageValues)
print(listOfAverageValues)


# #### Ranking Articles to a Cluster

# In[35]:


from scipy import spatial

rankArticlesToCluster=[]
for x in range(chosenNumberOfCluster):
    rankArticlesToCluster.append([])
    
for x in range (len(ArticleVector)):
    for index,value in enumerate(averageDistributionPerCluster):
        #finds cosine similarity between artlice vector and average vector of the cluster
        rankArticlesToCluster[index].append(tuple((x,1 - spatial.distance.cosine(ArticleVector[x], value))))
        
#sorting the ranked list
import operator
sortedRankArticlesToCluster=[]
for x in rankArticlesToCluster:
    sortedRankArticlesToCluster.append(sorted(x,key=lambda x: x[1]))

ascendingRankedArticlesToCluster = []
for x in sortedRankArticlesToCluster:
    ascendingRankedArticlesToCluster.append(list(reversed(x)))
        


# #### Ranking Clusters to an Article

# In[36]:


rankClustersToArticle = []
for x in range(len(ArticleVector)):
    rankClustersToArticle.append([])
    
for x in range(chosenNumberOfCluster):
    for index, value in enumerate(ArticleVector):
        rankClustersToArticle[index].append(tuple((x, 1-spatial.distance.cosine(value, averageDistributionPerCluster[x]))))
        
#sorting the ranked list

sortedRankClustersToArticle=[]
for x in rankClustersToArticle:
    sortedRankClustersToArticle.append(sorted(x,key=lambda x: x[1]))

ascendingRankClustersToArticle = []
for x in sortedRankClustersToArticle:
    ascendingRankClustersToArticle.append(list(reversed(x)))


# ## Metrics and Evaluation

# ### News Recommendation

# #### S@10 Version 1 where we compare if one user who posted the aricle exists in the community

# In[37]:


def sAt10OneUser():
    k=10
    total=0
    for x in range(len(ascendingRankedArticlesToCluster)):
        count = 0;
        for y in ascendingRankedArticlesToCluster[x][:k]:
            newsid = int(dfUniqueNewsId.iloc[[y[0]]].NewsId)
            for user in newsId2UserId[newsid]:
                if user in idsCluster[x]:
                    count+=10
                    total+=10
                    break
            if count != 0:
                break
    precisionVal = total/(chosenNumberOfCluster*10)
    print(precisionVal)


# #### S@10 Version 2 where we compare if all users who posted the aricle exists in the community
# 

# In[38]:


def sAt10AllUsers():
    k=10
    total=0
    for x in range(len(ascendingRankedArticlesToCluster)):
        count = 0;
        for y in ascendingRankedArticlesToCluster[x][:k]:
            newsid = int(dfUniqueNewsId.iloc[[y[0]]].NewsId)
            if len(set(newsId2UserId[newsid])&set(idsCluster[x])) == len(newsId2UserId[newsid]):
                count += 10
                total+=10
                break
    precisionVal = total/(chosenNumberOfCluster*10)
    print(precisionVal)


# In[39]:


sAt10OneUser()
sAt10AllUsers()


# #### MRR Version 1 where we compare if one user who posted the aricle exists in the community

# In[40]:


def mrrOneUser():
    mrr=0
    totalmrr = 0
    for x in range(len(ascendingRankedArticlesToCluster)):
        mrr = 0
        for index, y in enumerate(ascendingRankedArticlesToCluster[x]):
            newsid = int(dfUniqueNewsId.iloc[[y[0]]].NewsId)
            for user in newsId2UserId[newsid]:
                if user in idsCluster[x]:
                    mrr= (1/(index + 1))
                    totalmrr += mrr
                    break
            if(mrr != 0):
                break
    print(totalmrr/chosenNumberOfCluster)


# #### MRR Version 2 where we compare if all users who posted the aricle exists in the community
# 

# In[41]:


def mrrAllUsers():
    mrr=0
    totalmrr = 0
    for x in range(len(ascendingRankedArticlesToCluster)):
        mrr = 0
        for index, y in enumerate(ascendingRankedArticlesToCluster[x]):
            newsid = int(dfUniqueNewsId.iloc[[y[0]]].NewsId)
            if len(set(newsId2UserId[newsid])&set(idsCluster[x])) == len(newsId2UserId[newsid]):
                mrr= (1/(index + 1))
                totalmrr += mrr
                break
    print(totalmrr/chosenNumberOfCluster)


# In[42]:


mrrOneUser()
mrrAllUsers()


# ### User Prediction

# #### Precision and Recall

# In[43]:


NewsIdsKeys = list(newsId2UserId.keys())


# In[44]:


def precisionAndRecall():
    fn = 0
    tp = 0
    fp = 0
    precision = 0
    recall = 0
    
    for index, val in enumerate(NewsIdsKeys):
        fp = 0
        tp = 0
        fn = 0
        c = ascendingRankClustersToArticle[index][0][0]
        tp = len(set(newsId2UserId[val])&set(idsCluster[c]))
        fp = (len(idsCluster[c]) - tp)
        fn = (len(newsId2UserId[val]) - tp)

        if (tp+fp)!=0:
            precision = precision + tp/(tp+fp)
        if (tp+fn)!=0:
            recall = recall + tp/(tp+fn)
    overallPrecision = precision/len(NewsIdsKeys)
    overallRecall = recall/len(NewsIdsKeys)
    return (overallPrecision, overallRecall)


# In[45]:


precisionAndRecall()


# #### FMeasure

# In[46]:


x=precisionAndRecall()
fmeasure= 2*((x[0]*x[1])/(x[0]+x[1]))
print(fmeasure)


# #### TODO: STORE FINAL RESULTS FOR DIFF VALUES IN CSV AND SHOW TABLE 

# In[ ]:





# In[ ]:




