# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# https://github.com/rhosse/Team-Lyrical/blob/9b059145dc26dc4e2624c6b6147da01d1f51fcdd/data_lemmatization.py

# https://towardsdatascience.com/how-i-used-machine-learning-to-classify-emails-and-turn-them-into-insights-efed37c1e66

import spacy
import nltk
import gensim
#Gensim is an open-source library for unsupervised topic modeling.
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import spacy #To prepare text
import pyLDAvis #interactive topic model visualization
import pyLDAvis.gensim
import nltk #Natural Language Toolkit
import re #ex : "A" and "a"
from nltk.corpus import stopwords #to delete stop %matplotlib inline
import pandas as pd
import random
import numpy as np
import data
import matplotlib.pyplot as plt

def parse_raw_message(raw_message):
    lines = raw_message.split('\n')
    email = {}
    message = ''
    keys_to_extract = ['from', 'to']
    for line in lines:
        if ':' not in line:
            message += line.strip()
            email['body'] = message
        else:
            pairs = line.split(':')
            key = pairs[0].lower()
            val = pairs[1].strip()
            if key in keys_to_extract:
                email[key] = val
    return email

def map_to_list(emails, key):
    results = []
    for email in emails:
        if key not in email:
            results.append('')
        else:
            results.append(email[key])
    return results

def parse_into_emails(messages):
    emails = [parse_raw_message(message) for message in messages]
    return {
        'body': map_to_list(emails, 'body'),
        'to': map_to_list(emails, 'to'),
        'from_': map_to_list(emails, 'from')}


