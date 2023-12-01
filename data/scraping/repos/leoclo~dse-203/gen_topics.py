from distutils.command.clean import clean
import pandas as pd
import numpy as np
import re
from pprint import pprint

# Stanza
import stanza

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy
from spacy import displacy


import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import matplotlib.pyplot as plt
# %matplotlib inline # Only for jupyter notebook

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

# NLTK Stop words
from nltk.corpus import stopwords


class gen_topics():
    def __init__(self):
        self.stop_words = stopwords.words('english')
        self.movies_df = []

        # Initialize stanza
        self.nlp = stanza.Pipeline('en')
        self.nlp("don't won't")

        self.lda_model = [] # variable to hold the lda_model 
        self.df_mtopics = [] 
        self.df_topic_words = []


    def batch_process(self,
                      data_df,
                      batch_size = 30,
                      num_topics_per_batch = 10
                      ):
        total_topics = int(data_df.shape[0]/batch_size) - 1
        self.df_mtopics = pd.DataFrame(columns=['movie_id', 'topic'])

        try:
            for k in range(total_topics):
                df_movie_slice = data_df.iloc[k*batch_size:(k+1)*batch_size]
                if(not df_movie_slice.empty):
                    df_movie_topics = self.make_topics(df_movie_slice, num_topics=num_topics_per_batch)
                    df_movie_topics['topic'] = df_movie_topics['topic'] + k*num_topics_per_batch
                    self.df_mtopics = pd.concat( [self.df_mtopics, df_movie_topics[['movie_id', 'topic']]], axis=0)
                    if (k ==0):
                        self.df_topic_words = pd.DataFrame( [ [ k.split("*")[1] for k in x[1].split(" + ") ] for x in self.lda_model.show_topics()])
                    else:
                        self.df_topic_words = pd.concat( [self.df_topic_words, pd.DataFrame( [ [ k.split("*")[1] for k in x[1].split(" + ") ] for x in self.lda_model.show_topics()])], axis=0 ) 
        except:
            print("Failed at k = "+str(k))
            print(df_movie_slice)

        return (self.df_mtopics, self.df_topic_words)       

    def make_topics(self, movies_df, num_topics=10): # by default set the number of topics to 10

        self.movies_df = movies_df

        ## Drop movies with Nan
        self.movies_df['overview'].dropna(inplace=True)

        data = self.clean_data()
        
        # Convert sentences to words        
        data_words = list(self.sent_to_words(data))

        # Remove Stop Words
        data_words_nostops = self.remove_stopwords(data_words)

        self.nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

        # Do lemmatization keeping only noun, adj, vb, adv
        data_lemmatized = self.lemmatization(data_words_nostops, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

        # Create the dictionary and corpus

        id2word = corpora.Dictionary(data_lemmatized)

        # Create Corpus
        texts = data_lemmatized

        # Term Frequency list   
        corpus = [id2word.doc2bow(text) for text in texts]

        # Build LDA model
        self.lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                         id2word=id2word,
                                                         num_topics=num_topics, # setting the number of topics to 10
                                                         random_state=100,
                                                         update_every=1,
                                                         chunksize=100,
                                                         passes=10,
                                                         alpha='auto',                                                   
                                                         per_word_topics=True)  

        # Compute Perplexity
        # print('\nPerplexity: ', self.lda_model.log_perplexity(corpus))  
        # a measure of how good the model is. lower the better.

        # Compute Coherence Score
        # coherence_model_lda = CoherenceModel(model=self.lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
        # coherence_lda = coherence_model_lda.get_coherence()
        # print('\nCoherence Score: ', coherence_lda)

        # Add the topics column to the movies df
        self.movies_df['topic'] = [self.lda_model[corpus][text][0][0][0] for text in range(len(self.movies_df['overview']))]

        return self.movies_df

    def clean_data(self):
        # Convert to list
        data = self.movies_df.overview.values.tolist()        
        # Filters
        # Remove speical characters - except for , and -
        data = [re.sub('[^a-zA-Z0-9,@\- ]', '', str(sent)) for sent in data]

        # Remove Emails
        data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]

        # Remove new line characters
        data = [re.sub('\s+', ' ', sent) for sent in data]

        # Remove distracting single quotes
        data = [re.sub("\'", "", sent) for sent in data]

        # Remove empty strings and blank spaces

        data = [ x for x in data if (len(x.strip()) > 0)]

        # Convert everything to lower case

        data = [ str(x).lower() for x in data]

        return data

    def sent_to_words(self, sentences):
        for sentence in sentences:
            yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations
    
    # Functions for stopwords and lemmatization
    def remove_stopwords(self, texts):
        return [[word for word in doc if word not in self.stop_words] for doc in texts]


    def lemmatization(self, texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        """https://spacy.io/api/annotation"""
        texts_out = []
        for sent in texts:
            doc = self.nlp(" ".join(sent)) 
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        return texts_out

    # Results
    def print_topics(self):
        pprint(self.lda_model.print_topics())