import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
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


import os 
output = "books"
if not os.path.exists(output):
    os.makedirs(output)

lines = open("bible.txt", "r", encoding='utf-8').readlines()

books = [] 
books_idx = {}
# there are books which are First of, second of etc 
# we need to pick up their names 
multinames = ["Kings", "Paralipomenon", "Esdras", "Machabees",
              "Corinthians","Thessalonians", "Timothy", "Peter", "John"]
# then collect the name of the old testament books
old_books = ["Genesis", "Exodus", "Leviticus", "Numbers", "Deuteronomy","Josue", 
             "Judges", "Ruth", "1Kings", "2Kings", "3Kings", "4Kings", 
             "1Paralipomenon","2Paralipomenon", "1Esdras", "2Esdras", 
             "Tobias", "Judith", "Esther", "Job", "Psalms", "Proverbs", 
             "Ecclesiastes", "Canticle", "Wisdom", "Ecclesiasticus", "Isaias", 
             "Jeremias", "Lamentations", "Baruch", "Ezechiel", 
             "Daniel", "Osee", "Joel", "Amos", "Abdias", "Joans", "Micheas",
             "Nahum", "Habacuc", "Sophonias", "Aggeus", "Zacharias", "Malachias", 
             "1Machabees", "2Machabees"]

for i, val in enumerate(lines, 0):
    # retireve all the chapters 
    if "Chapter" in val:
        book_name = val.split()[0]
        possible_further_name = val.split()[1]
        if possible_further_name in multinames: 
            current_book_name = book_name + possible_further_name 
        else: 
            current_book_name = book_name
            
        if not current_book_name in books:        
            print(f"Adding {current_book_name} to books, starting idx {i}")
            if i==1:
                tmp_book = current_book_name 
            else:
                books_idx[tmp_book].append(i)
            tmp_book = current_book_name
            books.append(current_book_name)
            books_idx[current_book_name] = [i]


print(books_idx)

data = [] 
ifiles = glob.glob("books/*.txt")
for ifile in ifiles: 
    book = open(ifile, "r").read().strip() 
    data.append(book)

stop_words = stopwords.words('english')
stop_words.extend(["thy","thou","thee", "hath", "upon", "me", "him", "them", "shall","ye", "one", "unto", "us"])


def remove_stopwords(text, stop_words):
    outtext = ' '.join([word for word in text.split() if word not in stop_words])
    return outtext


for i, book in enumerate(data, 0):
    # remove NUMBER:NUMBER. pattern at the beginning
    data[i] = re.sub(r"\d{1,}\:\d{1,}\.", "",data[i])
    # remove NAME Chapter NUMBER 
    data[i] = re.sub(r"\w{1,} Chapter \d{1,}","",data[i] )
    #lower case 
    data[i] = data[i].lower() 
    # remove punctuation 
    data[i] = data[i].translate(str.maketrans('', '', string.punctuation))
    # remove new lines 
    data[i] = re.sub('\s+', " ", data[i]) 
    # remove new line
    data[i] = re.sub(r"\\n", " ", data[i])
    # remove stopwords 
    data[i] = ' '.join([word for word in data[i].split() if word not in stop_words]) #remove_stopwords(data[i], stop_words)
