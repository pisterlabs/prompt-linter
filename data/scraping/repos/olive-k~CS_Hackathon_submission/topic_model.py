import os
import numpy as np
import pandas as pd
import spacy
from spacy import displacy
import itertools as it
import re
from wordcloud import WordCloud
import gensim
from gensim.utils import simple_preprocess
import gensim.corpora as corpora
from gensim.models import CoherenceModel
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.corpus import stopwords
from pprint import pprint

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt
#%matplotlib inline
nlp = spacy.load('en_core_web_sm')


def topic_model():

    #Loading the dataset
    print("Loading the dataset!!")
    dir_path = "data/skytrax/skytrax_all.csv"
    fields = ['content']
    reviews_df = pd.read_csv(dir_path, usecols=fields)
    reviews_list = reviews_df.content.values.tolist()
    print("Dataset successfully loaded!!")
    
    #Preliminary preprocessing
    print("First stage of preprocrssing of the data started!!")
    reviews_list = [re.sub(r'\S*@\S*\s?', '', sent) for sent in reviews_list]
    reviews_list = [re.sub(r'\s+', ' ', sent) for sent in reviews_list]
    reviews_list = [re.sub(r"\'", "", sent) for sent in reviews_list]
    reviews_list = reviews_df.content.values.tolist()
    print("First stage of preprocrssing of the data completed!!")

    
    def sent_to_words(sentences):
        """tokenizes by creating a list of strings

        Args:
          List of reviews

        Returns:
          A generator

        """

        for sentence in sentences:
            yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))
    print("Second stage of preprocrssing of the data started!!")
    data_words = list(sent_to_words(reviews_list))

    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  

    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    print("Second stage of preprocrssing of the data completed!!")

    stop_words = stopwords.words('english')
    def remove_stopwords(texts):
        """Removes stopwords

        Args:
          List of texts
          
        Returns:
          A list

        """

        return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

    def make_bigrams(texts):

        """creates bigrams if any ex: new_york

        Args:
          List of lists of words
          
        Returns:
          A list

        """
        return [bigram_mod[doc] for doc in texts]

    def make_trigrams(texts):

        """creates trigrams if any ex: new_york_times

        Args:
          List of lists of words
          
        Returns:
          A list

        """
        return [trigram_mod[bigram_mod[doc]] for doc in texts]

    def lemmatization(texts, allowed_postags=['NOUN']):#, 'ADJ', 'VERB', 'ADV']):

        """lemmatizes or converts words to its base form and filters words
            according to allowed parts of speech

        Args:
          1.List of lists of words
          2.list of allowed parts of speech
          
        Returns:
          A list

        """
        """https://spacy.io/api/annotation"""
        texts_out = []
        for sent in texts:
            doc = nlp(" ".join(sent)) 
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        return texts_out

    # Remove Stop Words
    print("Final stage of preprocessing started!!")
    data_words_nostops = remove_stopwords(data_words)

    # Form Bigrams
    data_words_bigrams = make_bigrams(data_words_nostops)

    # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
    # python3 -m spacy download en
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

    # Do lemmatization keeping only noun, adj, vb, adv

    data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN'])#, 'ADJ', 'VERB', 'ADV'])
    print("Final stage of preprocessing completed!!")

    # Create Dictionary
    print("Buliding LDA model!!, Have some refreshments it will take some time!!")
    id2word = corpora.Dictionary(data_lemmatized)
    id2word.filter_extremes(no_below=20, no_above=0.4)
    id2word.compactify()

    # Create Corpus
    texts = data_lemmatized

    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]

    # Build LDA model
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                              id2word=id2word,
                                              num_topics=5, 
                                              random_state=100,
                                              update_every=1,
                                              chunksize=100,
                                              passes=75,
                                              alpha='auto',
                                              per_word_topics=True)



if __name__ == '__main__':
    topic_model()
    print("Following are the topics found!!")
    pprint(lda_model.print_topics())
    print("Buliding LDA model completed!!")