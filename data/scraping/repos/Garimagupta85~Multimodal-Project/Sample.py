import math
import random
import numpy as np
import scipy.sparse as sp
import pickle as pkl
import re
import time
import os
from math import log
from nltk.corpus import stopwords
import nltk
import pandas as pd
import requests
import pickle as pkl
import json
import pandas as pd
import gensim
import nltk
from nltk.util import ngrams
from nltk.collocations import BigramCollocationFinder
from nltk.stem.porter import PorterStemmer
from nltk.corpus import gutenberg
import multiprocessing
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim.parsing.preprocessing import stem_text
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from pprint import pprint
from gensim.models import Phrases
from gensim.models.phrases import Phraser
import string
from sklearn.metrics.pairwise import cosine_similarity
from functools import reduce

import cosinesim
from sentence_transformers import SentenceTransformer, util

nltk.download('stopwords') 
nltk.download('wordnet') 
nltk.download('punkt')

porter = PorterStemmer()

model = SentenceTransformer('paraphrase-MiniLM-L12-v2')
stop_words = stopwords.words('english')

def remove_emoji(string):
    emoji_pattern = re.compile("["
                            u"\U0001F600-\U0001F64F"  # emoticons
                            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                            u"\U0001F680-\U0001F6FF"  # transport & map symbols
                            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                            u"\U00002500-\U00002BEF"  # chinese char
                            u"\U00002702-\U000027B0"
                            u"\U00002702-\U000027B0"
                            u"\U000024C2-\U0001F251"
                            u"\U0001f926-\U0001f937"
                            u"\U00010000-\U0010ffff"
                            u"\u2640-\u2642"
                            u"\u2600-\u2B55"
                            u"\u200d"
                            u"\u23cf"
                            u"\u23e9"
                            u"\u231a"
                            u"\ufe0f"  # dingbats
                            u"\u3030"
                            "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)


def freq_with_cosine(text,fd1, word, threshold=0.2):
    """If cosine similarity meets the threshold, then increase frequence distribution
    Args:
        text ([string]): the whole document
        fdist ([type]): [description]
        word ([string]): candidate word
        threshold (float, optional): [description]. Defaults to 0.07.
    Returns:
        [int]: Updated frequency of the word
    """
    occurence = 1
    if word in stop_words:
        return occurence

    for w1 in fd1:
        #similarity = cosinesim.get_similarity_tfidf(model, stop_words, text, w1, word) 
        if w1 in stop_words:
            similarity = 0
        else:
            similarity = cosinesim.get_similartiy_bert(model, text, w1, word) #text: My school gave me an apple /w1: Apple/ word: Comptuer / compare Computer with text (with pretrained bert model)
        if similarity>threshold:
            occurence += 1
    return occurence

# print(striples['1'][0])

# def pmi(word1, word2, fd1, fd2, words, words1, words2, use_cosine):
#     # word1 = porter.stem(word1)
#     # word2 = porter.stem(word2)

#     word_freq_1 = fd1[word1] if fd1[word1]>0 else 1
#     word_freq_2 = fd1[word2] if fd1[word2]>0 else 1
#     print(f"Freq: {word_freq_1}, {word_freq_2} ")
#     if use_cosine:
#         word_freq_1 += freq_with_cosine(corpus[27].lower(),fd1, word1)
#         word_freq_2 += freq_with_cosine(corpus[27].lower(),fd1, word2)
#     print(f"Freq after Cosine: {word_freq_1}, {word_freq_2} ")
#     prob_word1 = word_freq_1/len(words1)
#     prob_word2 = word_freq_2/len(words2)
#     print(f"Prob: {prob_word1}, {prob_word2}")
#     prob_word1_word2 = word_freq_1 * word_freq_2 / (len(words1) * len(words2)) # 811*2; 811*811
#     print(f"ProbW1W2: {prob_word1_word2}")
#     pmi = math.log(prob_word1_word2/float(prob_word1*prob_word2), 2.71828)  # 0 for independence, and +1 for complete co-occurrence
#     print(f"PMI : {pmi}")
#     a = 1/len(words)
#     b = 1 - a * len(words)
#     npmi = a * pmi + b
#     #npmi = pmi/log(1.0 * count / num_window) -1
#     print(f"NPMI: {npmi:.4f}  ==> {word1}, {word2}")

#     if npmi> 0.01:
#         return True
#     return False
def meets_pmi(windows, words_len, word_1, word_2, uni_fdist, bi_fdist):
    # word_1 = porter.stem(word_1) #TODO: use lemmatization
    # word_2 = porter.stem(word_2)
    # #finder.ngram_fd[('this', 'bigram')]

    word1_parts = len(word_1.split(" "))
    fdist_w1 = uni_fdist if word1_parts ==1 else bi_fdist #TODO: check for trigrams
    #BigramCollocationFinder(word_1)
    word_freq_1 = fdist_w1[word_1] #fdist_w1[word_1] if fdist_w1[word_1] > 0 else 1
    word_freq_1 +=1 #to prevent divide by zero

    word2_parts = len(word_2.split(" "))
    fdist_w2 = uni_fdist if word2_parts ==1 else bi_fdist #TODO: check for trigrams
    word_freq_2 = fdist_w2[word_2] #fdist_w1[word_1] if fdist_w1[word_1] > 0 else 1
    word_freq_2+=1

    print("Frequencies before Cosine:")
    print(f"{word_1}: {word_freq_1} | {word_2}: {word_freq_2}" )

    #add to frequency based on the cosine similarity
    word_freq_1 += freq_with_cosine(sum(windows,[]),fdist_w1, word_1) #sum(windows,[]) flattens windows list
    word_freq_2 += freq_with_cosine(sum(windows,[]),fdist_w2, word_2)

    word1_word2_occurence = 1
    for window in windows:
        window_uni_fdist = nltk.FreqDist(window)  #frequency distribution of unigrams
        window_bigrams = ngrams(window, 2)
        window_bi_fdist = nltk.FreqDist(window_bigrams) #frequency distribution of bigrams

        
        window_fdist_w1 = window_uni_fdist if word1_parts ==1 else window_bi_fdist
        window_fdist_w2 = window_uni_fdist if word2_parts ==1 else window_bi_fdist

        # window_w1_occurence = window_fdist_w1[word_1]
        # window_w2_occurence = window_fdist_w2[word_2]        
        window_w1_occurence = freq_with_cosine(window, window_fdist_w1, word_1)
        window_w2_occurence = freq_with_cosine(window, window_fdist_w2, word_2)
        word1_word2_occurence += min(window_w1_occurence, window_w2_occurence)

    print(f"{word_1}: {word_freq_1} | {word_2}: {word_freq_2} | co-occurence: {word1_word2_occurence}" )
    # 0 for independence, and +1 for complete co-occurrence
    #pmi = math.log(word1_word2_occurence/(word_freq_1*word_freq_2), 2)
     
    num_window = len(windows)

    pmi = math.log((1.0 * word1_word2_occurence / num_window) / (1.0 * word_freq_1 * word_freq_2/(num_window * num_window))) #taken from VGCN paper
    npmi = math.log(1.0 * word_freq_1 * word_freq_2/(num_window * num_window))/math.log(1.0 * word1_word2_occurence / num_window) -1
    #TODO: entity categorization (check if they're in the same cateogory) one cateogry might be immediate parent, then it has more weight
    
    # a = 1/words_len
    # b = 1 - a * words_len
    # npmi = a * pmi + b
    #npmi = pmi/log(1.0 * count / num_window) -1
    # Check mehdi's 2016-17 paper
    print(f"PMI: {pmi}  ==> {word_1}, {word_2}")
    print(f"NPMI: {npmi:.4f}  ==> {word_1}, {word_2}")

    if npmi > 0.01:
        return True
    return False

striples = pd.read_csv("VGCN-BERT-master/filtered_triple_sample.csv")

words1 = list(striples['1']) #Tokens
print(len(words1))
words2 = list(striples['3']) #Objects from conceptnet
print(len(words2))
words = words1 + words2

data = [json.loads(l) for l in open('VGCN-BERT-master/ALONE-Toxicity-Dataset_v5_1.json', 'r')]
df = pd.DataFrame(data)
# print(df["Tweets"][687])
# print(str(reduce(lambda x,y: x+" "+y, df['Tweets'][26])))

# corpus = str(reduce(lambda x,y: x+" "+y, df['Tweets'][26]))

# corpus = []
Tweets = []
for i in range(len(df['Tweets'])):
    # for tweet in df['Tweets'][i]:
    # Tweet = [c if c not in map(str,range(0,10)) else "" for c in df['Tweets'][i]]
    Tweets.append(str(reduce(lambda x,y: x+""+y, df['Tweets'][i])))
        # corpus.append( str(reduce(lambda x,y: x+" "+y, Tweets)))
# print("\n")
# print(Tweets)
# print("\n")

print(len(Tweets))
print(Tweets[27])

# user_text = ' '.join(Tweets[27])
# # user_text= remove_emoji(user_text)
# user_text_tokens = nltk.tokenize.word_tokenize(user_text)
# uni_fdist = nltk.FreqDist(user_text_tokens)  #frequency distribution of unigrams
# bigrams = ngrams(user_text_tokens, 4)
# bi_fdist = nltk.FreqDist(bigrams) #frequency distribution of bigrams


window_size = 25 #TODO: make it word window size -- Make of tokens
list_of_windows = zip(*(iter(words1),) * window_size) 
windows = []
for window in list_of_windows:
    window_text = ' '.join(window)[0:100] #first words, to test
    window_text = remove_emoji(window_text)
    window_tokens = nltk.tokenize.word_tokenize(window_text)
    windows.append(window_tokens)

print(len(windows))

fd_one = nltk.FreqDist(words1) 
fd_two = nltk.FreqDist(words2)

Final_triples = []

for i in range(len(striples)):
    if meets_pmi(windows, len(words1), striples['1'][i], striples['3'][i], fd_one, fd_two): #windows, words_len, word_1, word_2, uni_fdist, bi_fdist
        Final_triples.append((striples['1'][i], striples['2'][i], striples['3'][i]))
    else:       
        # striples.drop(i, inplace = False)
        continue

# Final_triples = striples.reset_index(drop=True)

Final_triples = pd.DataFrame(Final_triples)

Final_triples.to_csv("VGCN-BERT-master/CosinePPmi.csv")