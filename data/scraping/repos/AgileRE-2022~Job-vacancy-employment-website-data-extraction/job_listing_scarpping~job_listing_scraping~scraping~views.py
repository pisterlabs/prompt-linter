# Django
from ast import For, keyword
from xml import dom
from django.http import HttpResponse
from matplotlib.pyplot import title
from . import urls

from django.http import HttpResponseRedirect
from django.shortcuts import render, redirect
import os
import numpy as np
import pandas as pd
import regex as re
from time import sleep

# Selenium
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support.ui import WebDriverWait
from django.contrib import messages

from parsel import Selector
from bs4 import BeautifulSoup

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

import regex as re


import gensim
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import gensim.corpora as corpora

from .parameter import getEmail,getPassword


def index(request):
    return render(request,"scraping/index.html")

def data(request):
    df=pd.read_csv('.\hasil.csv')
    df=df.to_dict()
    titles=df['Title']
    descriptions=df['Descriptions']
    array = []
    for i in titles:
        object = {
            'title' : titles[i],
            'description': descriptions[i]
        }
        array.append(object)
    context={
        'objects': array
    }
    print(titles)
    return render(request,'scraping/data.html',context)


def scrap(request):

    driver = webdriver.Edge(".\edgewebdriver\msedgedriver.exe")

    driver.get( 'https://www.linkedin.com')
    password_input = driver.find_element_by_name('session_password')
    password_input.send_keys(getPassword())
    username_input = driver.find_element_by_name('session_key')
    username_input.send_keys(getEmail())
    submit = driver.find_element_by_class_name('sign-in-form__submit-button')
    submit.click()
    sleep(10)
    driver.get('https://linkedin.com/jobs/search/?f_E=2&keywords=information%20systems')
    links = driver.find_elements_by_xpath("//a[@class='disabled ember-view job-card-container__link']")
    links =[link.get_attribute("href") for link in links]
    sleep(1)

    sleep(5)

    wait = WebDriverWait(driver, 20)
    links = driver.find_elements_by_xpath("//a[@class='disabled ember-view job-card-container__link']")
    links =[link.get_attribute("href") for link in links]
    sleep(1)

    
    listTitle = []
    listCompanies = []
    listLocations=[]
    listDescriptions=[]
    for link in links :
        driver.get(link)
        sleep(5)
        sel= Selector(text=driver.page_source)
        titles = sel.xpath('//h1[@class="t-24 t-bold jobs-unified-top-card__job-title"]/text()').extract()
        listTitle.append(titles[0])
        companies = sel.xpath('//span[@class="jobs-unified-top-card__company-name"]').extract()
        listCompanies.append(companies[0])
        locations = sel.xpath('//span[@class="jobs-unified-top-card__bullet"]/text()').extract()
        listLocations.append(locations[0])
        descriptions = sel.xpath('//*[@id="job-details"]').extract()
        listDescriptions.append(descriptions[0])

    #data cleaning
    listCompanies2=[]
    listLocations2=[]
    listDescriptions2=[]
    for company in listCompanies :
        company = BeautifulSoup(company,features='html.parser').text
        company = company.strip(' \n')
        listCompanies2.append(company)
    for location in listLocations :
        location = BeautifulSoup(location,features='html.parser').text
        location = location.strip(' \n')
        listLocations2.append(location)
    for description in listDescriptions :
        description = BeautifulSoup(description,features='html.parser').text
        description = re.sub(r'\n', '', description)
        listDescriptions2.append(description)

    #exporting to csv
    data = [listTitle,listCompanies2,listLocations2,listDescriptions2]        
    data = np.transpose(data)
    column = ['Title','Companies','Locations','Descriptions']
    df = pd.DataFrame(data=data, columns=column)
    df.to_csv('hasil.csv')

    driver.quit()
    messages.success(request, "Timetable extraction successful.")
    return render(request,"scraping/scraping.html")

def topicModeling(request):
    data=pd.read_csv('.\hasil.csv')
    data = data.drop(columns=['Companies','Locations'])
    stop_words = set(stopwords.words('english'))
    data['Descriptions_without_stopwords'] = data['Descriptions'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
    data['Descriptions_without_stopwords'] = data['Descriptions_without_stopwords'].str.replace('[^\w\s]', '')
    data['Descriptions_without_stopwords'] = data['Descriptions_without_stopwords'].map(lambda x: x.lower())
    data['Descriptions_without_stopwords'] = data['Descriptions_without_stopwords'].apply(lemmatize_text)
    dataLDA = data.Descriptions_without_stopwords.tolist()
    data_words = list(sent_to_words(dataLDA))
    data_words = remove_stopwords(data_words)

    #id2text
    id2word = corpora.Dictionary(data_words)
    texts = data_words
    corpus = [id2word.doc2bow(text) for text in texts]
    lda_model = gensim.models.ldamodel.LdaModel(corpus = corpus, id2word = id2word, num_topics=30, update_every=1, chunksize=100, passes=10, alpha="auto")
    df_topic_sents_keywords = format_topics_sentences(lda_model, corpus, data_words)

    # Format
    df_dominant_topic = df_topic_sents_keywords.reset_index()
    df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
    df_dominant_topic = df_dominant_topic.to_dict()
    dom_topic=df_dominant_topic['Dominant_Topic']
    keyword=df_dominant_topic['Keywords']
    data=data.to_dict()
    titles=data['Title']
    descriptions=data['Descriptions']
    array = []
    for i in titles:
        object = {
            'domtopic' : id2word[dom_topic[i]],
            'keyword' : keyword[i],
            'title' : titles[i],
            'description': descriptions[i]
        }
        array.append(object)
    context={
        'objects': array
    }
    return render(request,'scraping/result.html',context)

def sent_to_words (sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence),deacc=True))

def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    return ' '.join([lemmatizer.lemmatize(w) for w in text.split(' ')])

def remove_stopwords(texts):
    stop_words = set(stopwords.words('english'))
    return [[word for word in simple_preprocess(str(doc)) 
        if word not in stop_words] for doc in texts]

def format_topics_sentences(ldamodel, corpus, texts):
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