import joblib
#import ENG
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from gensim.models.coherencemodel import CoherenceModel
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
np.random.seed(2018)
import nltk
nltk.download('wordnet')

import time
import timeit

import pythainlp
from pythainlp import word_tokenize

import pandas as pd
from pandas import Series, DataFrame
import numpy as np

import os
# all function

#function to solve: ManagerFor, managerFor/ out = split words
#input = a word (only English)
#output = [w1, w2, w3,...]
import re
# all function

#function to solve: ManagerFor, managerFor
#input = a word (only English)
#output = [w1, w2, w3,...]
import re
def splitword(text):
  out=[]  # output: list of string
  couUp=0  # count of upper cases
  left=0  # left index
  initial=text.strip()
  upper=re.findall('[A-Z]',initial)  # find all upper cases
  for i in range(len(initial)):
    if initial[i] in upper:  # if char is upper case
      couUp+=1
    if couUp==len(upper):
      out.append(initial[left:i])
      out.append(initial[i:])
      break
    elif initial[i] in upper and i !=0:  # if char is upper case and not the first char
      out.append(initial[left:i])
      left=i
  return ' '.join(out).strip()

#Data Pre-processing for English only
# input = many words
#output = [w1, w2, w3, ......]

#1. Tokenization: Split the text into sentences and the sentences into words.
# Lowercase the words and remove punctuation.
#2. Words that have fewer than 3 characters are removed.
#3. All stopwords are removed.
#4. Words are lemmatized — 
#words in third person are changed to first person and verbs in past and future tenses are changed into present.
#5. Words are stemmed — words are reduced to their root form.
def lemmatize_stemming(text):
    return SnowballStemmer('english').stem(WordNetLemmatizer().lemmatize(text, pos='v'))
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result

#process for all data
#input = docs
# output = a list of each word
def preprocessAll(text):
  text=str(text)
  text=text.strip()
  thai=pythainlp.thai_characters  # all Thai chars
  th=''
  en=''
  for i in range(len(text)):
    if text[i] in thai:  # if char in Thai
      th=th+text[i]
    else:  # if char in Eng
      if (i != 0) and (text[i-1] in thai):  # if char is not the first char, and the previous char is Thai
        en=en+' '+text[i]
      else:  
        en=en+text[i]
  temp_en=[]  # record Eng chars
  for i in en.split(' '):
    if i !='':
      temp_en.append(splitword(i))
  temp_en=' '.join(temp_en)
  return preprocess(temp_en)+word_tokenize(th, keep_whitespace=False)

def predict_tfid_loadedModel(text, loaded_model):
  process_unseen= preprocessAll(text)
  dic_unseen=loaded_model.id2word
  bow_unseen=dic_unseen.doc2bow(process_unseen) 
  target=bow_unseen
  return sorted(loaded_model[target], key= lambda x:x[1], reverse= True)


#load test dataset
start = timeit.default_timer()
test=pd.read_csv('/home/std/Downloads/python/test_dataset_desPlusTitleV1_genRemoveV1_tag2_V9.csv')[['combine','job_label']]
test.dropna(inplace=True)
test.reset_index(drop=True, inplace=True)
job_des=test['combine']
job_label=test['job_label']


#load model
path='/home/std/Downloads/python/model_temp/Final_title_noBiTri_varyTopic_dictionaryV3_AlphaEtaV1/'
allModel=os.listdir(path)
raw=pd.DataFrame() #raw out
raw['tag_topic']=job_label
for model in sorted(allModel):
    loaded_model=joblib.load(path+model)
    raw_predict=[]
    for i in range(len(job_des)):
        t=predict_tfid_loadedModel(job_des.iloc[i], loaded_model)
        if len(t)==0:
            raw_predict.append(np.nan)
        else:
            raw_predict.append(t[0][0])
    raw[model]=pd.Series(raw_predict, index=raw['tag_topic'].index)

    
raw.to_csv('/home/std/Downloads/python/test_result/raw_result_Title_noBiTriFinalV3_AlphaBeta_tag2_V9'+'.csv',index=False, header=True)
stop = timeit.default_timer()
print('runtime_model '+model+': ', stop - start)
