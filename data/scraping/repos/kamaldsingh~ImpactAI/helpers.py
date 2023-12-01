import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import nltk
nltk.download('punkt')

from nltk.tokenize import sent_tokenize, word_tokenize
from transformers import BertTokenizer, BertForSequenceClassification, pipeline

import os
from datetime import datetime
from scipy.stats import kurtosis, skew
import itertools
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import randint
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)
import random
import re

from dateutil import parser
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain import PromptTemplate, LLMChain
from langchain.llms import HuggingFacePipeline
import torch
from transformers import pipeline

pd.options.display.max_columns = 999
import warnings
warnings.filterwarnings('ignore')
import gc
import re
from wordcloud import WordCloud, STOPWORDS
import ipywidgets as widgets
from ipywidgets import interact, interact_manual, interactive, fixed, interact_manual, Button, \
Box, VBox, HBox, Layout, FloatText, Textarea, Dropdown, Label, IntSlider, Button, GridBox, ButtonStyle, \
Combobox, BoundedFloatText, BoundedIntText, DatePicker, ToggleButton, Checkbox, FileUpload
from os.path import exists

stopwords = set(STOPWORDS)

#EDA
# from ydata_profiling import ProfileReport

finbert_esg = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-esg',num_labels=4)
tokenizer_esg = BertTokenizer.from_pretrained('yiyanghkust/finbert-esg')
esg_label_pip = pipeline("text-classification", model=finbert_esg, tokenizer=tokenizer_esg)

finbert_sentiment = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone',num_labels=3)
tokenizer_sentiment = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
sentiment_pipeline = pipeline("text-classification", model=finbert_sentiment, tokenizer=tokenizer_sentiment)

def parse_dt(s):
    try:
        return(parser.parse(str(s)))
    except:
        return(np.nan)

def get_esg_label_transcript(tr):
    sent_label_scores = []

    for sent in sent_tokenize(tr):
        all_esg_labels = esg_label_pip(sent)
        non_none_labels = [x for x in all_esg_labels if x['label']!='None']
        if(len(non_none_labels)>0):
            sent_label_scores.append([non_none_labels[0]['label'],non_none_labels[0]['score'],sent])
    df = pd.DataFrame(sent_label_scores, columns=['esg_label', 'label_score', 'sent'])
    return(df)

def create_sentiment_output(all_labels):
    non_none_labels = [x for x in all_labels if x['label']!='None']
    if(len(non_none_labels)>0):
        label = non_none_labels[0]['label']
        score = non_none_labels[0]['score']
        sentiment = 0
        if(label=='Positive'):
            return(1*score)
        elif(label=='Negative'):
            return(-1*score)
        else:
            return 0
    else:
        return 0
    
    
######  ESG ######
######  ESG ######
def generate_esg_cols(row,section='transcript'):
    if(row[section]!=np.nan):
        label_scores_df = get_esg_label_transcript(row[section])
        label_scores_df['sentiment'] = label_scores_df.sent.apply(lambda x: create_sentiment_output(sentiment_pipeline(x)))
        label_scores_df.sentiment = label_scores_df.sentiment.apply(lambda x:np.round(x,4))
        clean_scores = label_scores_df[((label_scores_df.label_score>0.7) & (label_scores_df.sentiment!=0))]
        group_senti = clean_scores.groupby('esg_label')['sentiment'].median().reset_index()
        for e in group_senti.esg_label.to_list():
            row[e+'_'+section] = group_senti[group_senti.esg_label==e].sentiment.iloc[0]
        missing_cols = list(set(cols_expect)-set(group_senti.esg_label.to_list()))
        # missing_cols = [c for c in cols_expect if c not in group_senti.esg_label.to_list()]
        for c in missing_cols:
            row[c+'_'+section] = np.nan
    else:
        for c in cols_expect:
            row[c+'_'+section] = np.nan
    return(row)


def generate_esg_cols_sections(tic):
    df = tp[tp.ticker==tic]
    try:
        df = df.apply(lambda x: generate_esg_cols(x,'prep_remarks'), axis = 1)
        df = df.apply(lambda x: generate_esg_cols(x,'QnA'), axis = 1)
    except:
        return(df)
    return(df)

def get_wordcloud(wordcloud_data_statements):
    comment_words = ''
    for val in wordcloud_data_statements:
        val = str(val)
        tokens = val.split()
        for i in range(len(tokens)):
            tokens[i] = tokens[i].lower()
        comment_words += " ".join(tokens)+" "
    
    wordcloud = WordCloud(width = 800, height = 800,
                background_color ='white',
                stopwords = stopwords,
                min_font_size = 10).generate(comment_words)
 
    # plot the WordCloud image                      
    plt.figure(figsize = (4, 4), facecolor = None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad = 0)
    plt.show()
    return
