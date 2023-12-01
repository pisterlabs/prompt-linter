from bs4 import BeautifulSoup
import requests
import re
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
import PyPDF2
import csv
import fitz
import time
import shutil
from IPython.display import HTML
from requests.exceptions import ConnectTimeout
import urllib.request
import PyPDF2
import io
import csv
import pytesseract
from pdf2image import convert_from_path
import shutil

import re
import os
import pandas as pd
import nlp

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

#Create new directory for given topic
# -------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------
def makeNewDiretoryForGivenTopic(userInputTopic):
    
    dirName = userInputTopic.replace(" ", "")

    parent_dir = dirName

    if os.path.exists(parent_dir):
        print("directory already exits")
        #shutil.rmtree(pdfDirName)
    else:
        path = os.path.join("", parent_dir)
        os.mkdir(path)
        print("Directory '% s' created" % parent_dir)

    pdfDirName = f"pdf{dirName}"
    paperDirName = f"paper{dirName}"
    imageDirName = f"image{dirName}"
    csvDirName = f"csv{dirName}"
    txtDirName = f"txt{dirName}"
    ldaDirName = f"lda{dirName}"

    directory_list = []
    directory_list.append(pdfDirName)
    directory_list.append(paperDirName)
    directory_list.append(imageDirName)
    directory_list.append(csvDirName)
    directory_list.append(txtDirName)
    directory_list.append(ldaDirName)
    
    # Create New Directories for given topic
    for dir_name in directory_list:
        if os.path.exists(f'{parent_dir}/{dir_name}'):
            print("directory already exits")
            #shutil.rmtree(pdfDirName)
        else:
            print(f'{parent_dir}/dir_name')
            path = os.path.join(parent_dir, dir_name)
            os.mkdir(path)
            print("Directory '% s' created" % dir_name)

    pdfDirName = f"{parent_dir}/pdf{dirName}"
    paperDirName = f"{parent_dir}/paper{dirName}"
    imageDirName = f"{parent_dir}/image{dirName}"
    csvDirName = f"{parent_dir}/csv{dirName}"
    txtDirName = f"{parent_dir}/txt{dirName}"
    ldaDirName = f"{parent_dir}/lda{dirName}"

    return pdfDirName, paperDirName, imageDirName, csvDirName, txtDirName, ldaDirName
###############################################################################################################
###############################################################################################################


# LDA Topic Modeling 
# -----------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------
# Convert sentences into words
def sentences_to_words(sentences):
    for sent in sentences:
        #convert sentence to word
        sent = gensim.utils.simple_preprocess(str(sent), deacc=True) 
        yield(sent)  

# Remove stopwords
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

# Lemmatized words into their root form
def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    texts_out = []
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

def make_bigrams(texts, bigram_mod):
    return [bigram_mod[doc] for doc in texts]

# Create LDA model 
def getLDAModel(textfile, ldaDirName, pdf_path):
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
    # Create Corpus: Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in data_lemmatized]
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
        lda_name = pdf_path.split('/')
        lda_name = lda_name[len(lda_name)-1]
        lda_name = f"{ldaDirName}/{lda_name}.html"
        vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word, mds="mmds", R=30)
        pyLDAvis.save_html(vis,lda_name)
        # Compute Perplexity
        print('\nPerplexity: ', lda_model.log_perplexity(corpus))  
        # a measure of how good the model is. lower the better.
        return lda_model, lda_name
    else:
        return None, ''
###############################################################################################################
###############################################################################################################


def getTopicKeywordsWithItsWeight(ldaModel):

    topic_list = ldaModel.show_topic(0, topn=30)
    res = dict()
    for x in topic_list:
        res[x[0]] = x[1]
    return res

def lda_model_listgetWeightList(new_pdf_path_list, new_text_path, new_lda_topic_key_word_list,lda_model_list, userInputTopic, lda_dir_name, new_image_list):


    userInputTopicKeyWord = userInputTopic.split()
    print(f"userInputTopicKeyWord: {userInputTopicKeyWord}")
    weight_list = []
    for y in range(0, len(lda_model_list)):
        keywordWithItsWeightList = getTopicKeywordsWithItsWeight(lda_model_list[y])
        print(keywordWithItsWeightList)

        weight = 0
        for x in userInputTopicKeyWord:
            result = keywordWithItsWeightList.get(x)
            if result == None:
                result = 0
            weight += result
        weight_list.append(weight)

    data = {'pdf_path': new_pdf_path_list, 'text_path': new_text_path, 'lda_topic_key_word_list' : new_lda_topic_key_word_list, 'lda_dir_name' : lda_dir_name, 'weight' : weight_list, 'images' : new_image_list}
    df = pd.DataFrame(data,columns=['pdf_path', 'text_path', 'lda_topic_key_word_list', 'lda_dir_name', 'weight', 'images'])
    new_list = df.sort_values(by=['weight'], ascending=False)
    return new_list
    


def getTopic(all_files, userInputTopic, ldaDirName):

    # Get the content of each papars and store them using a list
    text_content_list = []
    for txt in all_files.text_path:
        text = open(txt,'r').read()
        text_content_list.append(text)

    new_pdf_path_list = []
    lda_model_list = []
    new_lda_topic_key_word_list = []
    lda_dir_name = []
    new_image_list = []
    new_txt_path_list = []
    n = 0
    min_text = 200
    max_text = 1000000
    # Get topic for each paper
    for text in text_content_list:
        print(len(text))

        if (len(text) > min_text and len(text) < max_text):

            lda_model, lda_name = getLDAModel(text, ldaDirName, all_files.paper_path[n])
            if (lda_model != None):
                print(n)
                new_pdf_path_list.append(all_files.paper_path[n])
                lda_model_list.append(lda_model)
                lda_dir_name.append(lda_name)
                new_image_list.append(all_files.image_path[n])
                new_txt_path_list.append(all_files.text_path[n])
                topic = lda_model.show_topic(0, topn=30)
                
                p = ''
                for i in topic:
                    p = p + i[0] + "+" + str(i[1]) + " "
                    
                print(p)
                print(topic)
                new_lda_topic_key_word_list.append(p)
                print("added")
        else:
            print(f"{all_files.txt_name} is too large")
        n+=1
    
    new_list = lda_model_listgetWeightList(new_pdf_path_list, new_txt_path_list, new_lda_topic_key_word_list,lda_model_list, userInputTopic, lda_dir_name, new_image_list)
    
    return new_list
# -----------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------
def getLinkToPath(paper_with_topic):

    base_url = "http://127.0.0.1:5000/"
    paper_path_list = []
    text_path_list = []
    lda_dir_name_list = []
    weight_list = []
    images_list = []


    for path in paper_with_topic.pdf_path:
        paper_path_list.append(f'{base_url}{path}')

    for text in paper_with_topic.text_path:
        text_path_list.append(f'{base_url}{text}')

    for lda_name in paper_with_topic.lda_dir_name:
        lda_dir_name_list.append(f'{base_url}{lda_name}')

    for image in paper_with_topic.images:
        images_list.append(f'{base_url}{image}')

    new_paper_with_topic =  {'pdf_path': paper_path_list, 'text_path': text_path_list, 'lda_topic_key_word_list': paper_with_topic.lda_topic_key_word_list, 'lda_dir_name': lda_dir_name_list, 'images': images_list, 'weight':  paper_with_topic.weight}
    new_paper_with_topics = pd.DataFrame(new_paper_with_topic)
    for i in new_paper_with_topics:
        print(i)
    return new_paper_with_topics

def getListWithWeightedKeyWord(weight_pdf_list):
    new_pdf_path_list = []
    new_topic_key_word = []
    new_weight_list = []
    new_text_path = []
    new_image_path = []
    count = 0
    new_list = weight_pdf_list.sort_values(by=['weight'], ascending=False)
    for i in weight_pdf_list.weight:
        if (i > 0):
            print(i)
            new_pdf_path_list.append(f'http://127.0.0.1:5000/{new_list.pdf_path[count]}')
            print(f'new_pdf_path_list:{new_list.pdf_path[count]}')
            new_topic_key_word.append(new_list.lda_topic_key_word_list[count])
            print(f'new_topic_key_word:{new_list.pdf_path[count]}')
            new_text_path.append(f'http://127.0.0.1:5000/{new_list.text_path[count]}')
            print(f'new_text_path:{new_list.pdf_path[count]}')
            new_image_path.append(f'http://127.0.0.1:5000/{new_list.images[count]}')
            print(f'new_image_path:{new_list.pdf_path[count]}')
            new_weight_list.append(i)
            print(f'new_weight_list:{i}')
            print(f'count:{count}')
        count+=1

    new_list = {'pdf_path': new_pdf_path_list,'text_path' : new_text_path, 'lda_topic_key_word_list' : new_topic_key_word, 'images' : new_image_path, 'weight' : new_weight_list,}
    new_lists = pd.DataFrame(new_list)

    return new_lists


# Main method
# -------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------
def findTopicForPapers(userInputTopic):
    

    # # ----------------------------------------------------------------------------------------------------
    # Create new directory to save the files for given topic
    pdfDirName, paperDirName, imageDirName, csvDirName, txtDirName, ldaDirName = makeNewDiretoryForGivenTopic(userInputTopic)

    # ----------------------------------------------------------------------------------------------------
    # Search for the csv file generated by /extractInfoFromPapers
    paperList_textList_imageLists_csv = f'{csvDirName}/paperList_textList_imageList.csv'
    all_files = pd.read_csv(paperList_textList_imageLists_csv)

    # Find the topic of each paper and sort the list by their weight
    paper_with_topic = getTopic(all_files, userInputTopic, ldaDirName)
    # Save the info in a csv file
    paper_with_topic_csv = f'{csvDirName}/paper_with_topic.csv'
    paper_with_topic.to_csv(paper_with_topic_csv, quoting=csv.QUOTE_NONE, quotechar='',escapechar='\\')
    paper_with_topic_with_url = getLinkToPath(paper_with_topic)
    # Create a HTML file for the csv file for user to view
    paper_with_topic_html = f'{csvDirName}/paper_with_topic.html'
    HTML(paper_with_topic_with_url.to_html(paper_with_topic_html, render_links=True, escape=False)) 

    
    # ----------------------------------------------------------------------------------------------------   
    # Listing articles by their Weighted key word
    weight_pdf_list = pd.read_csv(paper_with_topic_csv)
    weighted_paper_with_topic = getListWithWeightedKeyWord(weight_pdf_list)
    weighted_paper_with_topic_csv = f'{csvDirName}/weighted_paper_with_topic.csv'
    weighted_paper_with_topic.to_csv(weighted_paper_with_topic_csv, quoting=csv.QUOTE_NONE, quotechar='',escapechar='\\')
    weighted_paper_with_topic_html = f'{csvDirName}/weighted_paper_with_topic.html'
    HTML(weighted_paper_with_topic.to_html(weighted_paper_with_topic_html, render_links=True, escape=False))  

    total_num_paper = len(paper_with_topic)
    num_paper_left = len(weighted_paper_with_topic)
    return_list  = []
    return_list.append(f'Number of paper processed: {total_num_paper}')
    return_list.append(f'Number of paper left: {num_paper_left}')

    table_name_list = []
    table_name_list.append(paper_with_topic_html)
    table_name_list.append(weighted_paper_with_topic_html)
    return return_list, table_name_list
    

###############################################################################################################
###############################################################################################################