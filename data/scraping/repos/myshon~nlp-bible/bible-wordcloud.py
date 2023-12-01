import re
from unicodedata import name
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
import glob
import string
import gensim
from gensim.corpora import Dictionary
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel, Phrases, LdaModel
from gensim.models.ldamulticore import LdaMulticore
import pandas as pd
from num2words import num2words
import numpy as np
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sbn 
import matplotlib.pyplot as plt
from scipy.stats import hmean
from scipy.stats import norm
from greek_accentuation.characters import *

import unidecode


import os 
output = "books"
if not os.path.exists(output):
    os.makedirs(output)

from pathlib import Path

data = [] 
book_names = []
ifiles = glob.glob("books/SBLGNTtxt/*.txt")
for ifile in ifiles: 
    book = open(ifile, "r", encoding='utf-8').read().strip() 
    name = Path(ifile).with_suffix('').name
    book_names.append(name)
    data.append(book)

def normalize_greek(word):
    word = remove_redundant_macron(strip_breathing(strip_accents(word)))
    return word.translate(str.maketrans('', '', string.punctuation)).lower()

stop_words = stopwords.words('greek')
stop_words_ext = [
"αὐτοῦ", "αὐτούς", "αὐτήν", "αὐτῇ", "αὐτῆ", "αὐτῆς", "αὐτῶν", "αὐτὸν", "αὐτόν", "αὐτόν" "αυτης", "αὐτοὶ", "αὐτοῖς", "τοῦτο", "τούτῳ", "αὐτῷ", "του","μου", "υμων", "εμοί", "ἐμοῦ", 
"εμού", "σου","σοῦ","σοὶ", "πάντα", "πολλούς", "εν", "ἡν", "υμείς", "καὶ", "Καὶ", "ἵνα", "ἐστὶν", "ὑμεῖς", "ἡμεῖς" "ὑμῖν", "ὑμᾶς", "ὑμῶν"]

# autou emin, tauta, eos, idou, eis, emas, pantes, moi, auto/w 
normalized = list(map(normalize_greek, stop_words_ext))
normalized = list(dict.fromkeys(normalized))

stop_words.extend(stop_words_ext)

#stop_words = list(map(unidecode.unidecode, stop_words))
stop_words = list(map(normalize_greek, stop_words))
# remove duplicates
stop_words = list(dict.fromkeys(stop_words))

for i, book in enumerate(data, 0):
    # remove NUMBER:NUMBER. pattern at the beginning
    text_index = data[i].find("\n")
    data[i] = data[i][text_index:]
    data[i] = re.sub(r"\w{1,} \d{1,}\:\d{1,}", "", data[i])
    # remove new lines 
    data[i] = re.sub('\s+', " ", data[i]) 
    # remove new line
    data[i] = re.sub("\n", " ", data[i])
    ## normalize word
    data[i] = normalize_greek(data[i])
    # remove stopwords 
    tokens = data[i].split()
    without_stopwords = ' '.join([word for word in tokens if word not in stop_words])
    data[i] = without_stopwords

all_books = ''
i = 1
for pbook in data:
    if i > 1: # only gospels first 4 books
        break
    all_books += pbook + " "
    i += 1
    
# WORDCLOUD
wordcloud = WordCloud(width=1600, height=800,max_font_size=200).generate(all_books)
plt.figure(figsize=(12,10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()