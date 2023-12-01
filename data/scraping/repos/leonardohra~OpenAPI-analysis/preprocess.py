# Import Dataset
import pandas as pd
import re
import gensim
from gensim.utils import simple_preprocess
from New_Gensim_Model import Gensim_Model
from nltk.corpus import stopwords
# spacy for lemmatization
import spacy
from CustomEnumerators import TopicModelingAlgorithm, CoherenceType
from pprint import pprint
from operator import itemgetter
from gensim.models.coherencemodel import CoherenceModel
import pickle


def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

def openapi_preprocess(data):
    # Remove html tags
    data = [re.sub('<[^>]*>', '', sent) for sent in data]
    data = [re.sub(r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)', '', sent) for sent in data]
    return data

def generate_bigrams_and_trigrams(data_words):
    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)

    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    return bigram_mod, trigram_mod

def buildstoplist(data_tk, threshold=0.1):
    index = {}
    wordcount = 0
    wordprob = {}
    stopwords = []
    for document in data_tk:
        for word in document:
            word_check = re.sub('[^a-zA-Z0-9 \n\.]', '', word)
            if word_check != '':
                wordcount += 1
                if word_check not in index:
                    index[word_check] = 1
                else:
                    index[word_check] += 1

    for word in index.keys():
        wordprob[word] = round(float(index[word])/wordcount*100,4)

    stoplist = sorted(wordprob.items(), key=itemgetter(1) ,reverse=True)

    for i in stoplist:
        if(i[1] >= threshold):
            stopwords.append(i[0])

    #stoplist contains tuples of the stop words ranked by the percentage of total word occurances
    #stopwords is a list of these words
    return stoplist, stopwords

def remove_stopwords(texts):
    stop_words = stopwords.words('english')
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(brigrams, texts):
    return [brigrams[doc] for doc in texts]

def make_trigrams(trigrams, bigrams, texts):
    return [trigrams[bigrams[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
    # python3 -m spacy download en
    nlp = spacy.load('en', disable=['parser', 'ner'])
    texts_out = []

    for sent in texts:
        doc = nlp(" ".join(sent))
        if(allowed_postags != None):
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        else:
            texts_out.append([token.lemma_ for token in doc])

    return texts_out

def execute_preprocessing(data_list, post=['NOUN', 'ADJ', 'VERB', 'ADV']):
    data_list = openapi_preprocess(data_list)
    data_words = list(sent_to_words(data_list))
    bigrams, trigrams = generate_bigrams_and_trigrams(data_words)
    # Remove Stop Words
    data_words_nostops = remove_stopwords(data_words)
    # Form Bigrams
    #data_words_bigrams = make_bigrams(bigrams, data_words_nostops)
    data_words_trigrams = make_trigrams(trigrams, bigrams, data_words_nostops)

    # Do lemmatization keeping only noun, adj, vb, adv
    data_lemmatized = lemmatization(data_words_trigrams, allowed_postags=post)
    #data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    #data_lemmatized = lemmatization(data_words_trigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

    return data_lemmatized

def execute_preprocessing_and_update_info(data_list, data_info, post=['NOUN', 'ADJ', 'VERB', 'ADV']):
    data_lemmatized = execute_preprocessing(data_list, post)

    lem_left = []
    new_info = {}
    id = 0
    #Remove empty documents
    for i in range(len(data_lemmatized)):
        if(data_lemmatized[i] != [] and not data_lemmatized[i] in lem_left):
            lem_left.append(data_lemmatized[i])
            new_info[id] = data_info[i]
            id += 1

    return lem_left, new_info
