#!/usr/bin/env python
# coding: utf-8

# In[13]:


#importing libraries that will be required

import nltk
import spacy
import gensim
import numpy as np
import pandas as pd
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import re
from pprint import pprint
import pickle

#for plotting a graph
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[14]:


#loading up the data set. this is a set of news articles from news websites set in the US, so they aren't completely in sync with India

news_datasets = pd.read_csv('articles1.csv')
news_datasets = news_datasets.head(2000)                  #top 2000 to save time
news_datasets.dropna()
news_datasets


# In[15]:


dataset = news_datasets[['title', 'content']]
dataset['title'] = dataset['title'].map(lambda st: ' '.join(st.split('-')[:-1]))
dataset['title'][4]


# In[16]:


#combining the title and content together for easy analysis
dataset['text'] = dataset['title'] + ' ' + dataset['content']


# In[17]:


#all the text is stored in data in the form of a numpy array
data = dataset.text.values


# In[18]:


#a function to convert the sentences to individual words
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

data_words = list(sent_to_words(data))
# to display a list of the words
#data_words


# In[21]:


#building a model for bigrams and trigrams
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=20) # higher threshold => fewer phrases.
#trigram = gensim.models.Phrases(bigram[data_words], threshold=60)  

bigram_mod = gensim.models.phrases.Phraser(bigram)
#trigram_mod = gensim.models.phrases.Phraser(trigram)

#print(trigram_mod[bigram_mod[data_words[0]]])


# In[22]:


bigram_mod.save('bigram_mod')


# In[9]:


#we start with data cleaning now. first define stopwords:
#NLTK Stop words
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
#stop_words.extend([])  input a list of words to add more stopwords depending on the data and the requirements


# In[10]:


# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


# In[11]:


# Remove Stop Words
data_words_nostops = remove_stopwords(data_words)

# Form Bigrams
data_words_bigrams = make_bigrams(data_words_nostops)

# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
nlp = spacy.load("en", disable=['parser', 'ner'])

# Do lemmatization keeping only noun, adj, vb, adv
data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

# to display the filtered out words after cleaning
#data_lemmatized


# In[ ]:


# Create Dictionary
id2word = corpora.Dictionary(data_lemmatized)

# Create Corpus
texts = data_lemmatized

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# View
#corpus[0:3]


# In[13]:


import os       #importing os to set environment variable
def install_java():
  get_ipython().system('apt-get install -y openjdk-8-jdk-headless -qq > /dev/null      #install openjdk')
  os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"     #set environment variable
  get_ipython().system('java -version       #check java version')
install_java()


# In[14]:


get_ipython().system('wget http://mallet.cs.umass.edu/dist/mallet-2.0.8.zip')
get_ipython().system('unzip mallet-2.0.8.zip')


# In[15]:


#trying to implement a mallet lda model
# Download File: http://mallet.cs.umass.edu/dist/mallet-2.0.8.zip

os.environ['MALLET_HOME'] = '/content/mallet-2.0.8'
mallet_path = '/content/mallet-2.0.8/bin/mallet' # you should NOT need to change this
#mallet_path = 'drive/My Drive/mallet/mallet-2.0.8/bin/mallet.bat' 
ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=28, id2word=id2word)


# In[ ]:


#find optimal numbers of topics

def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=5):
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
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=id2word)
        #model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=id2word,num_topics=20, random_state=100, update_every=1, chunksize=100, passes=20, alpha='auto', per_word_topics=True)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values


# In[ ]:


# Can take a long time to run.
# model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=data_lemmatized, start=8, limit=48, step=4)


# In[ ]:


# Show graph
# limit=48; start=8; step=4;
# x = range(start, limit, step)
# plt.plot(x, coherence_values)
# plt.xlabel("Num Topics")
# plt.ylabel("Coherence score")
# plt.legend(("coherence_values"), loc='best')
# plt.show()


# In[ ]:


# Print the coherence scores
# for m, cv in zip(x, coherence_values):
    #print("Num Topics =", m, " has Coherence Value of", round(cv, 4))


# In[29]:


# Select the optimal model and print the topics
# optimal_model = model_list[6]                 #keep the model having highest coherence
optimal_model = ldamallet
model_topics = optimal_model.show_topics(formatted=True)
pprint(optimal_model.print_topics(num_topics=28, num_words=20))


# In[17]:


def format_topics_sentences(ldamodel=optimal_model, corpus=corpus, texts=data):
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


df_topic_sents_keywords= format_topics_sentences(ldamodel=optimal_model, corpus=corpus, texts=data)

# Format
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

# Show
df_dominant_topic


# In[18]:


# Group top 5 sentences under each topic
sent_topics_sorteddf_mallet = pd.DataFrame()

sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')

for i, grp in sent_topics_outdf_grpd:
    sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet, 
                                             grp.sort_values(['Perc_Contribution'], ascending=[0]).head(1)], 
                                            axis=0)

# Reset Index    
sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)

# Format
sent_topics_sorteddf_mallet.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Text"]

# Show
sent_topics_sorteddf_mallet


# In[30]:


#to print text of a row
#pprint(sent_topics_sorteddf_mallet['Text'][25])


# In[31]:


#creating dictionary, has to be done manually
topic_map = {23.0: 'Education', 8.0: 'Russia in American Election', 20.0: 'Food and Restaurants', 0.0: 'Movies and Entertainment', 7.0: 'Law wrt Health and Insurance', 13.0: 'City Life', 1.0: 'American President Policy and Administration', 15.0: 'Election and Politics', 10.0: 'Crime and Police', 24.0: 'America-China Relationship', 27.0: 'Business Economy and Industry', 22.0: 'Fashion', 5.0: 'Military and War', 11.0: 'Medicine', 16.0: 'Science', 2.0: 'American Borders and Immigration', 3.0: 'Courts and Law', 4.0: 'Family Life', 6.0: 'Political Speech', 9.0: 'Sports', 12.0: 'Jobs', 14.0: 'American-British relations and Europe', 17.0: 'Travel', 18.0: 'Fitness and Lifestyle', 19.0: 'American Senate and Trump administration', 21.0: 'Internet and Technology', 25.0: 'Festivals and Celebrations', 26.0: 'Literature and Books'}
topic_map


# In[35]:


def top_map(num):
  return topic_map[num]
top_map(27)


# In[37]:


dataset['topic'] = df_dominant_topic.Dominant_Topic.apply(top_map)
dataset


# In[ ]:


dataset.to_pickle('data.pkl')


# In[ ]:




