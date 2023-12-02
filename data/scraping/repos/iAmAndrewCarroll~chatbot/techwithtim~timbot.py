
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy as np
print("tops")
import tflearn
# import tensorflow as tf
import random

import json
with open('intents.json') as file:
    data = json.load(file)

print("data: ", data)

import openai
from dotenv import load_dotenv
import os

# load env variables from .env
load_dotenv()

# retrieve the API key
api_key = os.getenv("GPT3_API_KEY")

words = []
labels = []
docs_x = []
docs_y = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent['tag'])

    if intent['tag'] not in labels:
        labels.append(intent['tag'])


