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

# ... (rest of your code)


data = [] 
ifiles = glob.glob("../../data/*.txt")
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
    
    
big_string = ""
for book in data:
    big_string += book + " "
    
output_file = "../../data/bible_corpus.txt"
with open(output_file, "w") as file:
    file.write(big_string)
    
# wordcloud = WordCloud(width=1600, height=800,max_font_size=200).generate(big_string)
# plt.figure(figsize=(12,10))
# plt.imshow(wordcloud, interpolation="bilinear")
# plt.axis("off")
# plt.show()