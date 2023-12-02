#DYNAMIC TOPIC MODEL
#Mai Vu
#May 2021


########################## IMPORT LIBRARIES ##########################

#Basic libraries
import time
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore", category = DeprecationWarning)

#Libraries for lemmatization
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

#Libraries for (dynamic) topic modeling
import gensim
from gensim import corpora
from gensim.models import CoherenceModel
from gensim.models import ldaseqmodel
from gensim.test.utils import datapath

print('- Imported libraries!')

########################## READ & PREPROCESS THE DATASET ##########################

###Read the data
eng_data = pd.read_csv('eng_abstracts_date+organization.csv')

#Create 2-year-interval time slices
eng_data.sort_values(by = ['date']) #Sort data according to the 'date' column
_, time_slices = np.unique(eng_data['date'], return_counts = True) #Count the number of abstracts in each year
# 2009 + 2010 + 2011 | 2012 + 2013 | 2014 + 2015 | 2016 + 2017 | 2018 + 2019
time_slices_2years_interval = [time_slices[0] + time_slices[1] + time_slices[2], time_slices[3] + time_slices[4],
                               time_slices[5] + time_slices[6], time_slices[7] + time_slices[8],
                               time_slices[9] + time_slices[10]]

print(time_slices)

###Preprocess data
eng_stopwords = set(stopwords.words('english')).union(gensim.parsing.preprocessing.STOPWORDS)
lemmatizer = WordNetLemmatizer()

#Tokenization and delete punctuation, number, short words and stop words
abstracts = []
for abstract in eng_data['abstract_en']:
    tokens = []
    for token in nltk.word_tokenize(abstract.lower()):
        if token.isalpha() and token not in eng_stopwords and len(token) > 3:
            tokens.append(lemmatizer.lemmatize(token))
    abstracts.append(tokens)

#Build the bigram
bigram = gensim.models.Phrases(abstracts, min_count = 10, threshold = 10)
for idx in range(len(abstracts)):
    abstracts[idx] = bigram[abstracts[idx]]

#Create dictionary for the given texts
dictionary = corpora.Dictionary(abstracts)
dictionary.filter_extremes(no_below = 10, no_above = 0.25) #Filter words that appear less than 10 documents and more than 25% of all documents

#Create the bag of words for all documents
bag_of_words = [dictionary.doc2bow(abstract) for abstract in abstracts]

print('- Read and preprocessed the dataset!')

########################## DYNAMIC TOPIC MODELING ##########################

#Build the model
print('- Training the model')
start_time = time.time() #Start count time
ldaseq = ldaseqmodel.LdaSeqModel(corpus = bag_of_words, id2word = dictionary, 
                                 time_slice = time_slices_2years_interval, num_topics = 8)
print('- Model finish running in', round((time.time() - start_time)/60), 'min(s)')
#Save the model
path = datapath('dynamic_model_code')
ldaseq.save(path)

########################## EVALUATION ##########################
coherence = ldaseq.dtm_coherence(time = 0)
temp = CoherenceModel(topics = coherence, corpus = bag_of_words, dictionary = dictionary, coherence = 'u_mass')
print ("u_mass = ", temp.get_coherence())
temp = CoherenceModel(topics = coherence, texts = abstracts, dictionary = dictionary, coherence='c_v')
print ("c_v = ", temp.get_coherence())