import pickle
import gensim
import numpy as np
import pandas as pd

from gensim.parsing.preprocessing import preprocess_string
from gensim.parsing.preprocessing import remove_stopwords
from gensim.parsing.preprocessing import strip_tags
from gensim.parsing.preprocessing import strip_punctuation

import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

#Now we are loading the pre-processed topic modelling component which is stored as a pickle file. 
df = pickle.load(open('/content/drive/My Drive/preprocessed wiki dump.pkl', "rb"))

# Create Dictionary
id2word = corpora.Dictionary(df['clean content'])

# Create Corpus
texts = df['clean content']

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# Build LDA model
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=100, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)

#To retrieve the toics for each of the document
doc_lda = lda_model[corpus]

#Retrieving topics for each of the document and appending into a list
document_topic_clean = []

for topics in doc_lda:
  document_topic_clean.append(topics)

#Function to sort the topics

def getKey(item):
    return item[1]
sorted_topic_list = []
working_sorted = []

#In this loop, I am looping each of the retrieved topics for each docment and retrieve only the top 2 topics per document

for topic_list in document_topic_clean:
  working_sorted = []
  working_sorted = sorted(topic_list, key=getKey)
  working_sorted.reverse()
  sorted_topic_list.append(working_sorted[:2])

#From each of the two topics, I am retrieving only the topic id values omitting the score. 
indi_topic_id = []
topic_id = []

for indi_elements in sorted_topic_list:
  for indi_topic in indi_elements:
    indi_topic_id.append(indi_topic[0])
  topic_id.append(indi_topic_id)
  indi_topic_id = []

#From each of the retrieved topic IDs, I retrieved the top two words. These words will be used as the keywords for each of the document
from progressbar import ProgressBar
pbar = ProgressBar()

individual_document_topic = [] # List to save the retrieved keywords for each of the article.
dump_document_topic = [] #List to store the lists of keywords for the entire dump. Each element represents a list which has the keywords for each of the document.

for pair in pbar(topic_id):
  for elements in pair:
    words = lda_model.show_topic(elements, topn=2)
    for indi in words:
      individual_document_topic.append(indi[0])
  dump_document_topic.append(individual_document_topic)
  individual_document_topic = []
  indi = []
  words = []

#Here, I appended the retrieved keywords and topic IDs into the dataframe
df['topics']=dump_document_topic

df['topic_id']=topic_id

#I saved the retrieved topic IDs along with the article title and content. This will be used to retrieve keywords for each of the articles.
pickle.dump(df, open( '/content/drive/My Drive/clean_topic_df.pkl', "wb" ) )

  

