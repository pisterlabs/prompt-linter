from typing import List
from hazm import *

import re
import numpy as np
import pandas as pd
from numpy import array

import convert_numbers

from gensim.corpora.dictionary import Dictionary
from nltk.util import ngrams
from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt
from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel

from wordcloud import WordCloud
from PIL import Image, ImageFont, ImageDraw
from bidi.algorithm import get_display
import arabic_reshaper
import matplotlib.pyplot as plt
import pyLDAvis.gensim

from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel

from nltk.util import ngrams

from wordcloud import WordCloud
from PIL import Image, ImageFont, ImageDraw
from bidi.algorithm import get_display
import arabic_reshaper
import matplotlib.pyplot as plt

import convert_numbers

from gensim.corpora.dictionary import Dictionary
import time

def read_files(filename):

  
  df = pd.read_csv(filename,encoding = "utf-8")

  stopwords = []
  with open(r'./my_stopwords.txt', 'r') as fp:
      for line in fp:
          x = line[:-1]
          stopwords.append(x)

  return df,stopwords


def normal(text):
    #normalize the text
    normalizer = Normalizer()
    text=str(text)
    text = normalizer.character_refinement(text)
    text = normalizer.punctuation_spacing(text)
    text = normalizer.affix_spacing(text)
    text = normalizer.normalize(text)
    return text

def remove_pattern(text):

    df_stopword = pd.read_csv("./Stopword/Final_Informal_dataframe_verb.csv")
    Informal_verb = df_stopword["verbs"].tolist()

    for j in Informal_verb:
      doc_string = re.sub(str(j), "", str(text))
    return doc_string


def remove_stopwords(stopwords,text):
    text=str(text)
    filtered_tokens = [token for token in word_tokenize(text) if token not in stopwords]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text

def remove_emoji(text): 
    emoji_pattern = re.compile("["
                u"\U0001F600-\U0001F64F"  # emoticons
                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                u"\U00002702-\U000027B0"
                u"\U000024C2-\U0001F251"
                u"\U0001f926-\U0001f937"
                u'\U00010000-\U0010ffff'
                u"\u200d"
                u"\u200c"
                u'\u200f'
                u'\u200e'
                u"\u2640-\u2642"
                u"\u2600-\u2B55"
                u"\u23cf"
                u"\u23e9"
                u"\u231a"
                u"\u3030"
                u"\ufe0f"

                u"\U000E006E|" \
                u"\U000E007F|" \
                u"\U000E0073|" \
                u"\U000E0063|" \
                u"\U000E0074|" \
                u"\U000E0077|" \
                u"\U000E006C"

    "]+", flags=re.UNICODE)
    
    return emoji_pattern.sub(r' ', text)

def remove_link(text): 
    return re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', str(text))

def remove_tabs(text): 
    return re.sub(r'[\n\r\t]', '', str(text))

def remove_email(text): 
    return re.sub(r'\S+@\S+', '', str(text))

def remove_englishword(text): 
    return re.sub(r'[A-Za-z0-9]+', '', str(text))

def remove_chars(text): 
    # return re.sub(r'\.(?!\d)', '', str(text))
    return  re.sub(r'[$+&+;+]|[><!+،:,\(\).+،+٬+,+]|[-+]|[…]|[\[\]»«//]|[\\]|[#+]|[_+]|[٪+]|[%]|[*+]|[؟+]|[?+]|[""]|@|' '|[٠|١|۴|۵|۶|۷|۸|۲|۳|۰|٤|٣|٢|۹|٨|٧|٦|۱|٥]', '', str(text))

def handle_delete_bad_chars(text):
    multiple_subs=[
        r'\u200f',
        r'\u200e',]
    for item in multiple_subs:
        doc_string = re.sub(item, '', str(text))
    return doc_string

def remove_extraspaces(text):
    return re.sub(r' +', ' ', text)

def remove_extranewlines(text):
    return re.sub(r'\n\n+', '\n\n', text)

def lemma(text):
    #lemmatize the text
    lemmatizer = Lemmatizer()
    text=str(text)
    return lemmatizer.lemmatize(text)

def preprocess(text):
    text = normal(text)
    # text = remove_stopwords(text)
    text = remove_emoji(text)
    text = remove_link(text)
    text = remove_tabs(text) 
    text = remove_email(text) 
    text = remove_englishword(text) 
    text = remove_chars(text)
    # text = lemma(text)
    text = remove_extraspaces(text) 
    text = remove_extranewlines(text) 
    return text

def ngram_convertor(sentence, stopwords, n=2, flag=1):
    ngram_list= []
    # tok= []
    if len(sentence)>1:
      if len(sentence.split())>1 : #sentence has more than one word
        if n== 2:
          sentence = remove_stopwords(stopwords,sentence)
        if len(sentence.split())>1  :   
          ngram_sentence = ngrams(sentence.split(), n)
          for tuple_ngram in ngram_sentence:
            ngram_list.append(" ".join(tuple_ngram))
        if flag == 1 :
          #Remove stopword in sentence
          sentence_pre = remove_stopwords(stopwords,sentence)
          if len(sentence_pre.split())>1  : 
            for item_tokens in word_tokenize(sentence_pre):
              if len(item_tokens)>1 and (item_tokens not in stopwords):
                ngram_list.append(item_tokens) #add tokens
        # ngram_list.append(tok)
    return ngram_list



def tokenize_ngram(stopwords,df,docs):
  my_docs =array(df['comment'].dropna())

  final_ngram_list= []
  for idx in range(len(my_docs)):
    flag= 1 #just one time tokenize
    for n in [2]: #,3]: #treegram
      sentence= my_docs[idx]
      # if len(sentence)>1 : #sentence not null
      # print(sentence) 
      final_ngram_list.append(ngram_convertor(sentence,stopwords,n,flag))
      flag= 0
    if len(final_ngram_list)>0 :
      docs[idx]= final_ngram_list
      final_ngram_list= []

  return docs



def final_corpus(docs):
  zx= []
  for i in range(len(docs)):
    try:
      if len(docs[i][0])>0 :
        zx.append(docs[i][0])
    except:
      print("*******************")

  # print(len(zx))
  dictionary = Dictionary(zx)

  dictionary.filter_extremes(no_below=1, no_above=1)
  #Create dictionary and corpus required for Topic Modeling
  corpus = [dictionary.doc2bow(zzz) for zzz in zx]

  return corpus,dictionary


def final_lda(corpus,dictionary,num_topic):
  # Set parameters.
  num_topics = num_topic
  chunksize = 500 #Number of Document to memory
  passes = 5 #how many time document can see 
  eval_every = 1  ####################
  iterations= 100

  temp = dictionary[0]  # only to "load" the dictionary.
  id2word = dictionary.id2token

  lda_model = LdaModel(corpus=corpus, id2word=id2word, chunksize=chunksize, \
                        alpha='auto', eta='auto', \
                        iterations=iterations, num_topics=num_topics, \
                        passes=passes, eval_every=eval_every)

  keys = []
  values= []



  case_list = []
  for i, topic in lda_model.show_topics(formatted=True, num_topics=10,num_words=10):
    # print(i, topic)
    case = {i:topic}
    case_list.append(case)

    topic = topic.split("+")
    for item in topic:
        item = item.split("\"")
        val = item[0].replace("*", "").replace(" ", "")
        try:
            value = float(val)
            values.append(value)
            keys.append(item[1])
        except: continue   

  return topic,keys,values,case_list


def create_dict(keys, values):
    topics = {}
    for key in set(keys):
        topics[key] = ([values[i] for i, value in enumerate(keys) if value == key])
    
    new_topics = {}
    for key in topics.keys():
        val = sum(topics[key])
        if val and len(key)>1:
            new_topics[key] = val  * 10000 
        else: continue
    return new_topics

def remove_unusful_topics(topics):
    return {x: y for x, y in topics.items() if not x.endswith("های") and not x.endswith("هایی")}

def remove_nested_keywords(topics):

    keys = list(topics.keys())
    values = list(topics.values())
    for i, sub_item in enumerate(keys):
        for j, item in enumerate(keys):
            if i==j:
                continue
            elif sub_item in item:
                keys[i] = "-1-1+1"
                values[i] = "-1-1+1"

    keys = list(filter(lambda a: a != "-1-1+1", keys))
    values = list(filter(lambda a: a != "-1-1+1", values))

    new = {}
    for i in range(len(keys)):
        try:
            new_key = keys[i]
            if len(new_key.split())>1 and new_key.split().count("_") > 1 and not new_key.split()[-1]!="_" and not new_key.split()[0] != "_":
                new_key = '\u200c'.join(new_key.split())
            if len(new_key) > 3:
                new[new_key] = values[i]
        except: continue
    return new



def wordcloud(topics):
        image_address = "./word_cloud.png"
        mask = np.array(Image.open("./mask-instagram.png"))
        wordcloud = WordCloud(max_font_size=80, background_color="white", font_path="./Vazir-Bold.ttf",
                              max_words=80, mask=mask, 
                              margin=10, height=800, width=800, colormap="Dark2", prefer_horizontal=1)
        #new_topics = {}
      
        #for x, y in topics.items():
            #try:
                #new_topics[get_display(arabic_reshaper.reshape(x))] = y
            #except: continue
        # print(new_topics)
        # topics = new_topics
        wordcloud.generate_from_frequencies(topics)
        wordcloud.to_file(image_address)
        image = Image.open(image_address)
        image.thumbnail((800, 800), Image.Resampling.LANCZOS)
        image = image.save(image_address, 'png', quality=100)
        # plt.imshow(wordcloud)
        plt.show()
        
        return topics


def run(filename,num_topic):

  df,stopwords= read_files(filename)

  col_number= len(df.columns)
  if col_number>2 :
    print("Your dataframe should only have 1 column.")
  else:
    df.columns.values[0] = 'comment'
    # df.rename(columns={ df.columns[1]: "" }, inplace = True)

    print("preprocessing started")
    start_pre = time.time()

    df['comment'] = df['comment'].apply(preprocess)
    df['comment'] = df['comment'].dropna()

    #remove each row of dataframe with len_word <2
    df['comment']= pd.DataFrame(df[df['comment'].map(len) > 2]["comment"])
    final_df = pd.DataFrame(df['comment'].dropna()).reset_index()

    docs =array(final_df['comment'])
    end_pre = time.time() - start_pre
    print("preprocessing finished: " + str(end_pre) + "\n")
    ########
    start_lda = time.time()
    print("lda started")
    docs= tokenize_ngram(stopwords,final_df,docs)
    corpus,dictionary= final_corpus(docs)
    topic,keys,values,case_list= final_lda(corpus,dictionary,num_topic)
    end_lda = time.time() - start_lda
    print("lda finished: " + str(end_lda) + "\n")
    ########
    topics= create_dict(keys, values)
    topics= remove_unusful_topics(topics)
    # topics= remove_nested_keywords(topics)
    wordcloud(topics)

    return case_list
# run(filename="./3_filimo.csv",num_topic=5)
#run(filename="Instagram_comment_server.csv",num_topic=5)
# run(filename="namava.csv",num_topic=5)
####################################################
from fastapi import FastAPI, status
from pydantic import BaseModel


app = FastAPI()


class UserIn(BaseModel):
    filename: str
    topic_num: int #EmailStr

@app.post("/")
def create_user(user: UserIn):
    case_list= run(filename=user.filename,num_topic=user.topic_num)
    return case_list
#######################################






