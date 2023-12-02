from bs4 import BeautifulSoup
import requests
import re
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
import PyPDF2
import csv
import pandas as pd
import time
import os
import shutil
from IPython.display import HTML
from requests.exceptions import ConnectTimeout
import urllib.request
import PyPDF2
import io
import re
import pandas as pd
import csv
import os
import pytesseract
from pdf2image import convert_from_path
import shutil

import re
import os
import numpy as np
import pandas as pd
import nlp
import csv

# Gensim
import gensim, spacy, logging, warnings
import gensim.corpora as corpora
from gensim.models import CoherenceModel
from nltk.stem.porter import *
# spacy for lemmatization
import spacy
import pyLDAvis
import pyLDAvis.gensim_models  # don't skip this
from nltk.corpus import stopwords
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
import nltk
import pprint
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're','in','for', 'and', 'of','the', 'is', 'edu', 'to', 'use', 'not', 'would', 'say', 'could', '_', 'be', 'know', 'good', 'go', 'get', 'do', 'done', 'try', 'many', 'some', 'nice', 'thank', 'think', 'see', 'rather', 'easy', 'easily', 'lot', 'lack', 'make', 'want', 'seem', 'run', 'need', 'even', 'right', 'line', 'even', 'also', 'may', 'take', 'come'])

def cleanText(text):
    text = text.lower()
    #remove references
    pos = text.rfind('references')
    text = text[:pos]
    text = text.replace(r'-', '')
    text = text.replace(r'_', '')
    text = text.replace(r')', '')
    text = text.replace(r'(', '')
    text = text.replace(r':', '')
    text = text.replace(r'?', '')
    text = text.replace(r'%', '')
    text = text.replace(r'+', '')
    text = text.replace(r'=', '')
    text = text.replace(r'&', '')
    text = text.replace(r'^', '')
    text = text.replace(r'$', '')
    text = text.replace(r'#', '')
    text = text.replace(r'~', '')
    text = text.replace(r'`', '')
    text = text.replace(r'[', '')
    text = text.replace(r']', '')
    text = text.replace(r'{', '')
    text = text.replace(r'}', '')
    text = text.replace(r'\\', '')
    text = text.replace(r'|', '')
    text = text.replace(r';', '')
    text = text.replace(r'<', '')
    text = text.replace(r'>', '')
    text = text.replace(r'"', '')
    text = text.replace(r'*', '')
    text = text.replace(r'/', '')
    
    text = re.sub(r'\S*@\S*\s?', '', text)  # remove emails
    text = re.sub(r'\s+', ' ', text)  # remove newline chars
    text = re.sub(r"\'", "", text)  # remove single quotes
    text = re.sub(r"http[s]?\://\S+","",text) #removing http:
    text = re.sub(r"[0-9]", "",text) # removing number
    text = re.sub(r'\s+',' ',text) # removing space
    text = text.encode('ascii', 'ignore').decode()
    
    return text

def extractTextFromScannedPDFs(fname):
    text = ""
    while True:
        #code to extract text from scanned pdfs
        pages = convert_from_path(fname)
        try:
            for page in pages:
                text += pytesseract.image_to_string(page)
            new_text = text.strip("\n")
        except:
            print("Oops!  PDF not readable")
            break
    return new_text


def extractTextFromPDFs(fname):
    #code to extract text from pdfs

    text = ""
    while True:
        try:
            pdfFileObj = open(fname, 'rb')
            pdfReader = PyPDF2.PdfReader(pdfFileObj,strict=False)
            text = ""
            print(f'page{pdfReader.numPages}')
            if (pdfReader.numPages > 2):
                for i in range(pdfReader.numPages):
                    pageObj = pdfReader.getPage(i)
                    text += pageObj.extractText()
                pdfFileObj.close()
                break
            break
        except:
            print("Oops!  PDF not readable")
            break

    return text

def getText(link):

    download = False  
    text = "" 
    while (download == False):
        try:
            print('Sending request')

            req = urllib.request.Request(link, headers={'User-Agent' : "Magic Browser"})
            remote_file = urllib.request.urlopen(req, timeout=20).read()
            remote_file_bytes = io.BytesIO(remote_file)
            pdfdoc = PyPDF2.PdfFileReader(remote_file_bytes)

            for i in range(pdfdoc.numPages):
                current_page = pdfdoc.getPage(i)
                print("===================")
                print("Content on page:" + str(i + 1))
                print("===================")
                print(current_page.extractText())
                print(len(current_page.extractText()))
                text += current_page.extractText()
            download = True
            print('data recived')
            text = cleanText(text)
            print("Cleaned Text")
            print(text)
        except:
            print('Request has timed out, retrying')
    
    return text
# -----------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------


# LDA Topic Modeling 
# -----------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------
# data cleaning
def sentences_to_words(sentences):
    for sent in sentences:
        #convert sentence to word
        sent = gensim.utils.simple_preprocess(str(sent), deacc=True) 
        yield(sent)  

# remove stopwords
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

# lemmatized words into their root form
def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    texts_out = []
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

def make_bigrams(texts, bigram_mod):
    return [bigram_mod[doc] for doc in texts]

def getLDATopic(textfile, lda_model_name):
    data = []
    data.append(textfile)
    sentences = data
    data_words = list(sentences_to_words(sentences))

    bigram = gensim.models.Phrases(data_words, min_count=1, threshold=100) # higher threshold fewer phrases.
    bigram_mod = gensim.models.phrases.Phraser(bigram)

    # Remove Stop Words
    data_words_nostops = remove_stopwords(data_words)

    data_words_bigrams = make_bigrams(data_words_nostops, bigram_mod)
    # Do lemmatization keeping only noun, adj, vb, adv
    data_lemmatized = lemmatization(data_words_nostops, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    # Create Dictionary
    id2word = corpora.Dictionary(data_lemmatized)

    print('dictionary')
    for i in id2word:
        print(i)

    # Create Corpus: Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in data_lemmatized]

    for i in corpus:
        print(i)
    # Build LDA model
    if (len(data_lemmatized[0]) > 5):
        lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=id2word,
                                                num_topics=1, 
                                                random_state=100,
                                                update_every=1,
                                                chunksize=10,
                                                passes=10,
                                                alpha='symmetric',
                                                iterations=100,
                                                per_word_topics=True)
        vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word, mds="mmds", R=30)
        lda_model_name = f'lda_model/{lda_model_name}.html'
        pyLDAvis.save_html(vis, lda_model_name)
        # Compute Perplexity
        print('\nPerplexity: ', lda_model.log_perplexity(corpus))  
        # a measure of how good the model is. lower the better.
        return lda_model_name
    else:
        return None
# -----------------------------------------------------------------------------------------------------

# Main method
# -----------------------------------------------------------------------------------------------------
def gettopic(userInputArticleLink):
    
    # # Download the link
    # # # #----------------------------------------------------------------------------------------------------
    # # Read the links from the csv file
    
    text = getText(userInputArticleLink)
    
    name = userInputArticleLink.split('/')
    lda_model_name = name[(len(name)-1)]
    lda_model_name = getLDATopic(text, lda_model_name)

    return_list = []
    return_list.append(lda_model_name)

    return return_list
