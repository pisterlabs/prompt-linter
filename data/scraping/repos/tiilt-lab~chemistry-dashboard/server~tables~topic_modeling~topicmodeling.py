import os
import re
import numpy as np
import pandas as pd
from pprint import pprint

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt


import PyPDF2

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

from nltk.corpus import stopwords
stop_words = stopwords.words('english')


mallet_path = 'mallet-2.0.8/bin/mallet' # update this path
fileurl = "testing/"
def add_stop_words(words):
    stop_words.extend(words)
    
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations
        
def generate_bigram(data_words):
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    
    return [bigram_mod[doc] for doc in data_words]


def generate_trigram(data_words, bigram):
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    
    return trigram, trigram_mod

def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
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
            model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=dictionary)
            model_list.append(model)
            coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
            coherence_values.append(coherencemodel.get_coherence())

        return model_list, coherence_values

    

def process_file(file_url):
    data = []
    if file_url.endswith(".pdf"):
        pdf = PyPDF2.PdfFileReader(open(file_url, "rb"))
        for page in pdf.pages:
            text = page.extractText()
            text.rstrip('\n')
            data.append(text)
    else:       
        with open (file_url, "r") as myfile:
            #add the line without any newline characters
            for line in myfile:
                currentLine = line.rstrip('\n')
                if currentLine != "" and currentLine != " ":
                    data.append(currentLine)
                
    data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]

    # Remove new line characters
    data = [re.sub('\s+', ' ', sent) for sent in data]

    # Remove distracting single quotes
    data = [re.sub("\'", "", sent) for sent in data]
    
    return data;

def generate_corpus(file_url, extra_stop_words):
    stop_words = ['from', 'subject', 're', 'edu', 'use']
    if extra_stop_words:
        stop_words = stop_words + extra_stop_words
    
    add_stop_words(stop_words)
    
    data = []
    if (os.path.isdir(file_url)):
        for subdir, dirs, files in os.walk(file_url):
            for file in files:
                filepath = subdir + os.sep + file
                if filepath.endswith(".txt") or filepath.endswith(".pdf"):
                    data = data + process_file(filepath)
    else:
        data = process_file(file_url)
    
    
    data_words = list(sent_to_words(data))
    stop_words = ['from', 'subject', 're', 'edu', 'use']
    if extra_stop_words:
        stop_words = stop_words + extra_stop_words
    
    add_stop_words(stop_words)
    
    def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        """https://spacy.io/api/annotation"""
        texts_out = []
        for sent in texts:
            doc = nlp(" ".join(sent)) 
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        return texts_out

    # Remove Stop Words
    data_words_nostops = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in data_words]

    # Form Bigrams
    data_words_bigrams = generate_bigram(data_words_nostops)

    # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
    # python3 -m spacy download en
    nlp = spacy.load('en', disable=['parser', 'ner'])

    # Do lemmatization keeping only noun, adj, vb, adv
    data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

    #print(data_lemmatized[:1])

    # Create Dictionary
    id2word = corpora.Dictionary(data_lemmatized)

    # Create Corpus
    texts = data_lemmatized

    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]
    
    return id2word, texts, corpus
    
def generate_topic_model(id2word, texts, corpus, number_of_topics):
    # View
    #print(corpus[:1])

    [[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]]

    lda_model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=number_of_topics, id2word=id2word)
    
    ldamallet = lda_model
    # Show Topics
    pprint(ldamallet.show_topics(formatted=False))

    # Compute Coherence Score
    coherence_model_ldamallet = CoherenceModel(model=ldamallet, texts=texts, dictionary=id2word, coherence='c_v')
    coherence_ldamallet = coherence_model_ldamallet.get_coherence()
   # print('\nCoherence Score: ', coherence_ldamallet)
    
    return lda_model
    # Visualize the topics

def find_optimal_num_topics(id2word, data_lemmatized, corpus):
    # Can take a long time to run.
    model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=data_lemmatized, start=2, limit=40, step=6)

    # Show graph
    limit=40; start=2; step=6;
    x = range(start, limit, step)
    #plt.plot(x, coherence_values)
    #plt.xlabel("Num Topics")
    #plt.ylabel("Coherence score")
    #plt.legend(("coherence_values"), loc='best')
    #plt.show()

    # Print the coherence scores
    maxCoherence = float("-inf")
    maxM = None
    for m, cv in zip(x, coherence_values):
        print("Num Topics =", m, " has Coherence Value of", round(cv, 4))
        if cv > maxCoherence:
            maxCoherence = cv
            maxM = m
        
    return maxM

def visualize_topic_model(lda_model, corpus, id2word):
    pyLDAvis.enable_notebook()
    vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
    vis


