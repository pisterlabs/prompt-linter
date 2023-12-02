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

from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from gensim.models.doc2vec import TaggedDocument
import PyPDF2
from nltk.stem import WordNetLemmatizer 
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
    # text = text.replace(r':', '')
    # text = text.replace(r'?', '')
    text = text.replace(r'%', '')
    text = text.replace(r'+', '')
    text = text.replace(r'=', '')
    text = text.replace(r'&', '')
    text = text.replace(r'^', '')
    text = text.replace(r'$', '')
    text = text.replace(r'#', '')
    # text = text.replace(r'!', '')
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
    #text = re.sub(r"[0-9]", "",text) # removing number
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
        except:
            print('Request has timed out, retrying')
    
    return text
# -----------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------------------------
# Doc2Vec
def get_taggeddoc(txt_list):
    
    #read all the documents under given folder name
    doc_list = txt_list
    
    tokenizer = RegexpTokenizer(r'\w+')
    en_stop = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()
 
    taggeddoc = []
 
    texts = []
    for index,i in enumerate(doc_list):
        
        # for tagged doc
        wordslist = []
        tagslist = []
 
        # clean and tokenize document string
        raw = i.lower()
        tokens = tokenizer.tokenize(raw)
        #print(tokens)
 
        # remove stop words from tokens
        stopped_tokens = [i for i in tokens if not i in en_stop]
        #print(stopped_tokens)
        #print("--------------------------------------------------------------")
        # remove numbers
        number_tokens = [re.sub(r'[\d]', ' ', i) for i in stopped_tokens]
        number_tokens = ' '.join(number_tokens).split()

        #print(number_tokens)
        #print("--------------------------------------------------------------")
        # stem tokens
        lemmatized_tokens = [lemmatizer.lemmatize(i) for i in number_tokens]
        #print(lemmatized_tokens)
        #print("--------------------------------------------------------------")
        
        # remove empty
        length_tokens = [i for i in lemmatized_tokens if len(i) > 1]
        # add tokens to list
        lemmatized_tokens.append(",")

        texts.append(lemmatized_tokens)
        
 
        td = TaggedDocument(gensim.utils.to_unicode(str.encode(' '.join(lemmatized_tokens))).split(),str(index))
        taggeddoc.append(td)
        #print(taggeddoc)
 
    return taggeddoc


def getDec2Vec(similar_article_list, similar_article_txtDir_list):
    # build the model
    #model =  gensim.models.doc2vec.Doc2Vec(taggeddoc, vector_size=30, min_count=1, epochs=80)
    txt_list = []
    for txt in similar_article_txtDir_list:
        text = open(txt,'r').read()
        txt_list.append(text)

    taggeddoc = get_taggeddoc(txt_list)


    model =  gensim.models.doc2vec.Doc2Vec(taggeddoc, vector_size=30, min_count=1, epochs=80)
    #model.build_vocab(taggeddoc)
    #model.train(taggeddoc, total_examples=model.corpus_count, epochs=model.epochs)
    similarity_matric = []

    for i in range(0, len(taggeddoc)):
        
        for j in range(0, len(taggeddoc)):
            temp_similarity_matric = []
            temp_similarity_matric.append(similar_article_list[i])
            temp_similarity_matric.append(similar_article_list[j])
            temp_similarity_matric.append(model.dv.similarity(i, j))
            similarity_matric.append(temp_similarity_matric)
    

    return similarity_matric
# Main method
# -----------------------------------------------------------------------------------------------------
def getsimilarity(userInputArticleLink1, userInputArticleLink2):
    
    # # Download the link
    # # # #----------------------------------------------------------------------------------------------------
    # # Read the links from the csv file
    text_list = []
    text_list.append(getText(userInputArticleLink1))
    text_list.append(getText(userInputArticleLink2))

    taggeddoc = get_taggeddoc(text_list)
    model =  gensim.models.doc2vec.Doc2Vec(taggeddoc, vector_size=30, min_count=1, epochs=80)
    #model.build_vocab(taggeddoc)
    #model.train(taggeddoc, total_examples=model.corpus_count, epochs=model.epochs)
    similarity = model.dv.similarity(0, 1)
    print(similarity)
    similarity_list = []
    similarity_list.append(f'{similarity}')

    return similarity_list
