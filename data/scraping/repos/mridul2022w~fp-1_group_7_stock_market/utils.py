import pandas as pd
import numpy as np
import requests
from pprint import pprint
import stockstats
import nltk
from bs4 import BeautifulSoup as bs
import re
import os
from sklearn import feature_extraction
import autocorrect
from autocorrect import Speller
import unidecode
import langdetect
from langdetect import detect
import deep_translator
from deep_translator import GoogleTranslator
import matplotlib.pyplot as plt
import gensim.corpora as corpora
import gensim
from gensim.utils import simple_preprocess
import wordcloud
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import seaborn as sns
from datetime import date
from nsepy import get_history
import snscrape.modules.twitter as sntwitter
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk import sent_tokenize, word_tokenize
import time
from gensim.parsing.preprocessing import strip_punctuation, strip_tags, strip_numeric
from nltk.stem.wordnet import WordNetLemmatizer   
import string
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
import math
from statsmodels.tools.eval_measures import rmse
import statsmodels.api as sm
import itertools
from statsmodels.tsa.arima_model import ARIMA, ARMA
import warnings
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense
from datetime import date,datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from keras.callbacks import EarlyStopping
import datetime
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
import pickle
from sklearn import linear_model
nltk.download('punkt')

st.set_page_config(layout="wide")
analyzer = SentimentIntensityAnalyzer()
lemma = WordNetLemmatizer()
warnings.filterwarnings("ignore")

#expanding the contractions
CONTRACTION_MAP = {
"ain't": "is not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't.": "could not.",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he'll've": "he he will have",
"he's": "he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how is",
"i'd": "i would",
"i'd've": "i would have",
"i'll": "i will",
"i'll've": "i will have",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'd've": "it would have",
"it'll": "it will",
"it'll've": "it will have",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she would",
"she'd've": "she would have",
"she'll": "she will",
"she'll've": "she will have",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so as",
"that'd": "that would",
"that'd've": "that would have",
"that's": "that is",
"there'd": "there would",
"there'd've": "there would have",
"there's": "there is",
"they'd": "they would",
"they'd've": "they would have",
"they'll": "they will",
"they'll've": "they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what'll've": "what will have",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"when's": "when is",
"when've": "when have",
"where'd": "where did",
"where's": "where is",
"where've": "where have",
"who'll": "who will",
"who'll've": "who will have",
"who's": "who is",
"who've": "who have",
"why's": "why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you would",
"you'd've": "you would have",
"you'll": "you will",
"you'll've": "you will have",
"you're": "you are",
"you've": "you have",
}


#Get stock id and name from stockedge.com
def search_stock_get_id(sym):
    abc = "NA"
    
    url  = "https://api.stockedge.com/Api/SecurityDashboardApi/GetMajorSecurities?term="+sym+"&page=1&pageSize=10&lang=en"
    response = requests.request("GET", url)
    df = pd.DataFrame(response.json())
    
    l = df['ID'].iloc[0]
    url = "https://api.stockedge.com/Api/SecurityDashboardApi/GetLatestSecurityInfo/"+str(l)+"?lang=en"
    response = requests.request("GET", url)
    resp = response.json()
    
    #get stock id
    if 'Type' in resp:
        if resp['Type'] == "Equity":
            if resp['Listings']:
                for j in range(0,len(resp['Listings'])):
                    if resp['Listings'][j]['ExchangeName'] == 'NSE':
                        name = resp['Name'].replace("Ltd.", "Limited")
                        abc = {'ID':resp['Listings'][j]['ListingID'],'short_name':resp['Slug'],"NAME":name,'SYMBOL':sym,'sym_id':l}
    
    #get index id
    if 'Type' in resp:
        if resp['Type'] == "Index":
            if resp['Listings']:
                for j in range(0,len(resp['Listings'])):
                    if resp['Listings'][j]['ExchangeName'] == 'NSE':
                        name = resp['Name'].upper()
                        abc = {'ID':resp['Listings'][j]['ListingID'],'short_name':resp['Slug'],"NAME":name,'SYMBOL':sym,'sym_id':l}
    
    

    return abc


#Get news data from groww    
def get_data_from_grow(inp):
    df12 = pd.DataFrame()
    for i in range(1,10):
        URL="https://groww.in/v1/api/stocks_company_master/v1/company_news/groww_contract_id/"+inp+"?page="+str(i)+"&size=100"
        a = requests.get(URL)
        df12 = df12.append(pd.DataFrame(a.json()['results']))
    
    if df12.shape[0]:
        df12['Date'] = df12['pubDate'].copy()
        df12 = df12[['title','summary','url','Date','source']]
        df12['Date'] = pd.to_datetime(df12['Date']).dt.date
        df12['Date'] = pd.to_datetime(df12['Date'])
        df12.sort_values(by='Date', ascending=False)
    
    return df12


#get news data from stockedge
def get_news_data_from_stock_edge(sym,n=10):
    
    a = search_stock_get_id(sym)
    numb = a['sym_id']
    df = pd.DataFrame()
    for i in range(1,n):
        URL="https://api.stockedge.com/Api/SecurityDashboardApi/GetNewsitemsForSecurity/"+str(numb)+"?page="+str(i)+"&pageSize=100&lang=en"
        a = requests.get(URL)
        df = df.append(pd.DataFrame(a.json()))
    if df.shape[0]:
        df['Date'] = pd.to_datetime(df['Date'],format='%Y-%m-%d')
        df.sort_values(by='Date', ascending=False)
        df = df[['Date','Description']]
        df['text'] = df[['Description','Date']].groupby(['Date'])['Description'].transform(lambda x: ','.join(x))
        df = df[['Date','text']].drop_duplicates()
    
    return df


#get stock ohlc data from stockedge
def get_stock_prices_ohlc(sym):
    
    sbin = get_history(symbol=sym,start=date(2017,1,1),end=date(2022,1,1))
    sbin = sbin.reset_index()
    sbin = sbin[['Date','Symbol', 'Series', 'Prev Close', 'Open', 'High', 'Low', 'Last','Close', 'VWAP', 'Volume']]
    if sbin.shape[0]:
        sbin['Date'] = pd.to_datetime(sbin['Date'],format='%Y-%m-%d')
        sbin.sort_values(by='Date', ascending=False)
    return sbin

def get_stock_prices_ohlc_modified(sym,start,end):
    
    sbin = get_history(symbol=sym,start=start,end=end)
    sbin = sbin.reset_index()
    sbin = sbin[['Date','Symbol', 'Series', 'Prev Close', 'Open', 'High', 'Low', 'Last','Close', 'VWAP', 'Volume']]
    if sbin.shape[0]:
        sbin['Date'] = pd.to_datetime(sbin['Date'],format='%Y-%m-%d')
        sbin.sort_values(by='Date', ascending=False)
    return sbin


#get fundatmental data from stockedge
def get_fundamental_data_from_stock_edge(numb):
    
    df1 = pd.DataFrame()
    URL="https://api.stockedge.com/Api/SecurityDashboardApi/GetCompanyEquityInfo/"+str(numb)+"/2?lang=en"
    a = requests.get(URL)
    df1 = pd.DataFrame([a.json()])
    
    return df1


#calculate indicators
def calculate_stock_indicators(stockprices,type,columnName):
    ''' Stock Indicators '''

    stock = stockstats.StockDataFrame.retype(stockprices)
    stock = stock[type]
    stock = stock.reset_index().rename(columns={type:columnName})
    new1 = pd.merge(stockprices, stock, left_on='date',right_on='date')

    return new1


#get data from twitter
def get_data_from_twitter(sym,n=10000,sdate='2017-01-01',edate='2022-01-01'):
    
    Tweet_limits = n
    tweet_list = [] #Creating list to append tweet data to
    users_name = ['livemint','ReutersIndia','moneycontrolcom','NDTVProfit','EconomicTimes','safalniveshak','BMEquityDesk','forbes_india','bsindia','ETMarkets','FinancialTimes','BloombergQuint','Investopedia','CNBCTV18Live','IIFL_Live','ZeeBusiness','BT_India','NSEIndia','BT_India','BloombergTV','WSJMarkets']
    for n, k in enumerate(users_name):
        for i, tweet in enumerate(sntwitter.TwitterSearchScraper("#"+sym+" from:"+users_name[n]+" since:"+sdate+" until:"+edate).get_items()):
            print(Tweet_limits)
            if i>Tweet_limits:
                break
            tweet_list.append({'Date':tweet.date, 'id':tweet.id, 'content':tweet.content, 'username':tweet.username})
        
    df = pd.DataFrame(tweet_list)
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    df['Date'] = pd.to_datetime(df['Date'],format="%Y-%m-%d")
    df.sort_values(by='Date', ascending=False)
    df['twitter_text'] = df[['content','Date']].groupby(['Date'])['content'].transform(lambda x: ','.join(x))
    df = df[['Date','twitter_text']].drop_duplicates()
    
    return df

def get_data_from_twitter_upd(sym,n=10000,sdate='2017-01-01',edate='2022-01-01'):
    
    Tweet_limits = n
    tweet_list = [] #Creating list to append tweet data to
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper("#"+sym+" since:"+sdate+" until:"+edate).get_items()):
        print(Tweet_limits)
        if i>Tweet_limits:
            break
        tweet_list.append({'Date':tweet.date, 'id':tweet.id, 'content':tweet.content, 'username':tweet.username})
        
    df = pd.DataFrame(tweet_list)
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    df['Date'] = pd.to_datetime(df['Date'],format="%Y-%m-%d")
    df.sort_values(by='Date', ascending=False)
    df['twitter_text'] = df[['content','Date']].groupby(['Date'])['content'].transform(lambda x: ','.join(x))
    df = df[['Date','twitter_text']].drop_duplicates()
    
    return df

#remove newlines,tab spaces
def remove_newlines_tabs(text):
    Formatted_text = text.replace('\\n', ' ').replace('\n', ' ').replace('\t',' ').replace('\\', ' ').replace('. com', '.com')
    return Formatted_text


#remove html tags
def rev_html_Tag(text):
    soup = bs(text, "html.parser")
    new_text=soup.get_text(separator=" ")
    return new_text


#remove links from text
def rev_link(text):
    rev_link=re.sub(r'http\S+','',text)
    rev_com=re.sub(r'\[A-Za-z]*\.com','',rev_link)
    return rev_com


# removing any whitespace
def rev_whitespace(text):
    pattern=re.compile(r'\s+')
    text_new=re.sub(pattern,' ',text)
    return text_new


#replacing accented character with alphabat
def accented_chatacter(text):
    k=re.sub(r"\x92", "'",text)
    return k


#removing addiotional accented characters from text
def rev_asc(text):
    text=unidecode.unidecode(text)
    return text


# converting to lower case
def lower_case(text):
    text=text.lower()
    return text


#removing repeated charactor
def rev_rep(text):
    
    Pattern_alpha = re.compile(r"([A-Za-z])\1{1,}", re.DOTALL)
    Formatted_text = Pattern_alpha.sub(r"\1\1", text)
    Pattern_Punct = re.compile(r'([.,/#!$%^&*?;:{}=_`~()+-])\1{1,}')
    Combined_Formatted = Pattern_Punct.sub(r'\1', Formatted_text)
    Final_Formatted = re.sub(' {2,}',' ', Combined_Formatted)
    
    return Final_Formatted


#removing puncutations
def rev_puc(text):
    
    punc='''!"#$%&'()*+,-/:;<=>?@[\]^_`{|}~'''
    Formatted_text=re.sub(r"[^a-zA-Z0-9.]+",' ',text)
    Formatted_text2=re.sub(r"[()]+",' ',Formatted_text)
    
    return Formatted_text2


#map above dict with the text
def string_contraction_mapping(text):
    
    list_of_tokens = text.split(' ')
    for Word in list_of_tokens:
        if Word in CONTRACTION_MAP:
            list_of_tokens = [item.replace(Word, CONTRACTION_MAP[Word]) for item in list_of_tokens]
    String_Of_tokens = ' '.join(str(e) for e in list_of_tokens)
    
    return String_Of_tokens


#detecting the language
def language_detection(text):
    try:
        l=detect(text)
    except Exception as E:
        l='en'
    
    return l


#translating other languages to English
def changing_language(k,n):
    if n != 'en':
        translated = GoogleTranslator(source=n, target='en').translate(k)
    else:
        translated = k
    return translated


#checking the spelling
def spell_check(text):
    spell=Speller(lang='en')
    Corrected_text = spell(text)
    return Corrected_text


#removing all $ signs
def rev_dollar(text):
    rev_dollar=re.sub(r"\$|'",'',text)
    return rev_dollar


# defining unit func to process one doc
def vader_unit_func(doc0):
    sents_list0 = sent_tokenize(doc0)
    vs_doc0 = []
    sent_ind = []
    for i in range(len(sents_list0)):
        vs_sent0 = analyzer.polarity_scores(sents_list0[i])
        vs_doc0.append(vs_sent0)
        sent_ind.append(i)
        
    # obtain output as DF    
    doc0_df = pd.DataFrame(vs_doc0)
    doc0_df.insert(0, 'sent_index', sent_ind)  # insert sent index
    doc0_df.insert(doc0_df.shape[1], 'sentence', sents_list0)
    return(doc0_df)


#function to call all docs
def vader_wrap_func(col,corpus0):
    
    # use ifinstance() to check & convert input to DF
    if isinstance(corpus0, list):
        corpus0 = pd.DataFrame({'text':corpus0})
    
    # define empty DF to concat unit func output to
    vs_df = pd.DataFrame(columns=['doc_index', 'sent_index', 'neg', 'neu', 'pos', 'compound', 'sentence'])    
    
    # apply unit-func to each doc & loop over all docs
    for i1 in range(len(corpus0)):
        doc0 = (corpus0[col].iloc[i1])
        vs_doc_df = vader_unit_func(doc0)  # applying unit-func
        vs_doc_df.insert(0, 'doc_index', i1)  # inserting doc index
        vs_df = pd.concat([vs_df, vs_doc_df], axis=0)
        
    return(vs_df)


#getting data from different source and merging it
def data_fetch_and_merge_from_different_sources(list_of_stocks):
    start = time.time()
    df_final = pd.DataFrame()
    
    for i in list_of_stocks:
        
        #fetch news data
        df = get_news_data_from_stock_edge(i)
        
        #fetch ohlc
        df1 = get_stock_prices_ohlc(i)
        
        #fetch data from twitter
        df2 = get_data_from_twitter(i)
        
        #merge
        df_with_news = df1.merge(df,on='Date',how='left')
        df_final = df_final.append(df_with_news.merge(df2,on='Date',how='left'))
    
    #saving data incase of faliure
    df_final.to_csv('data/data.csv')
    print("Time Taken (sec): ",(time.time())-start)
    
    return df_final


#cleaning the data
def clean_the_data(data,col_names):
    
    start = time.time()
    
    data_des_1 = data.copy()
    for i in col_names:
        updated_col = ['Symbol','Date',i]
        data_des = data_des_1[updated_col]
        data_des=data_des.dropna()
        data_des['Description2']=data_des[i].apply(lambda text:remove_newlines_tabs(text))
        data_des['Description3']=data_des['Description2'].apply(lambda text:rev_html_Tag(text))
        data_des['Description3']=data_des['Description2'].apply(lambda text:rev_link(text))
        data_des['Description4']=data_des['Description3'].apply(lambda text:rev_whitespace(text))
        data_des['Description5']=data_des['Description4'].apply(lambda text:accented_chatacter(text))
        data_des['Description6']=data_des['Description5'].apply(lambda text:rev_asc(text))
        data_des['Description7']=data_des['Description6'].apply(lambda text:lower_case(text))
        data_des['Description8']=data_des['Description7'].apply(lambda text:rev_rep(text))
        data_des['Description9']=data_des['Description8'].apply(lambda text:rev_puc(text))
        data_des['Description10']=data_des['Description9'].apply(lambda text:string_contraction_mapping(text))
        data_des['language']=data_des['Description10'].apply(lambda text:language_detection(text))
        data_des['Description11']=data_des.apply(lambda text:changing_language(text['Description10'],text['language']),axis=1)
        data_des['Description12']=data_des['Description11'].apply(lambda text:string_contraction_mapping(text))
        
        #data_des['Description13']=data_des['Description12'].apply(lambda text:spell_check(text))
        data_des[i+'_Description14']=data_des['Description12'].apply(lambda text:rev_dollar(text))
        
        data_des = data_des[['Date',i+'_Description14','Symbol']]

        data=pd.merge(data,data_des,on=["Date",'Symbol'],how="left")
        
    data.to_csv('data/cleaned_data.csv')
    
    print("Time Taken (sec): ",(time.time())-start)
    
    return data


#getting the compound score
def sentiment_analysis(data,col_names):
    start = time.time()
    
    data1 = data.copy()
    for i in col_names:
        col = [i+"_Description14",'Date','Symbol']
        data_des4 = data[col]
        data_des4 = data_des4.dropna()
        col1 = i+"_Description14"
        data_des2_vs=vader_wrap_func(col1,data_des4)
        sentiment = data_des2_vs.groupby('doc_index').sum()
        sentiment = sentiment.reset_index()
        sentiment = sentiment.add_prefix(i+'_')
        data_des4 = data_des4[['Date','Symbol']]
        data_des4 = data_des4.reset_index(drop=True)
        data12 = pd.concat([data_des4,sentiment],axis=1)
        data1 = data1.merge(data12,on=['Date','Symbol'],how='left')
    
    data1.to_csv('data/data_with_sentiment.csv')
    
    print("Time Taken (sec): ",(time.time())-start)
    
    return data1


#
def textClean(text0):
    text1 = [[word for word in ''.join(doc).split()] for doc in text0]
    normalized = [[" ".join([word for word in ' '.join(doc).split()])] for doc in text1]
    return normalized


# obtain the factor matrices - beta
def build_beta_df(lda_model, id2word):
    beta = lda_model.get_topics()  # shape (num_topics, vocabulary_size).
    beta_df = pd.DataFrame(data=beta)

    # convert colnames in beta_df 2 tokens
    token2col = list(id2word.token2id)
    beta_df.columns = token2col
    # beta_df.loc[0,:].sum()  # checking if rows sum to 1

    # convert rownames too, eh? Using format(), .shape[] and range()
    rowNames=['topic' + format(x+1, '02d') for x in range(beta_df.shape[0])]
    rowNames_series = pd.Series(rowNames)
    beta_df.rename(index=rowNames_series, inplace=True)
    return(beta_df)


# func to get gamma matrix by looping using list.comp
def build_gamma_df(lda_model, corpus0,id2word):
    gamma_doc = []  # empty list 2 populate with gamma colms
    num_topics = lda_model.get_topics().shape[0]
    
    for doc in range(len(corpus0)):
        doc1 = corpus0[doc].split()
        bow_doc = id2word.doc2bow(doc1)
        gamma_doc0 = [0]*num_topics  # define list of zeroes num_topics long
        gamma_doc1 = lda_model.get_document_topics(bow_doc)
        gamma_doc2_x = [x for (x,y) in gamma_doc1]#; gamma_doc2_x
        gamma_doc2_y = [y for (x,y) in gamma_doc1]#; gamma_doc2_y
        for i in range(len(gamma_doc1)):
            x = gamma_doc2_x[i]
            y = gamma_doc2_y[i]
            gamma_doc0[x] = y  # wasn't geting this in list comprehension somehow 
        gamma_doc.append(gamma_doc0)
        
    gamma_df = pd.DataFrame(data=gamma_doc)  # shape=num_docs x num_topics
    topicNames=['topic' + format(x+1, '02d') for x in range(num_topics)]
    topicNames_series = pd.Series(topicNames)
    gamma_df.rename(columns=topicNames_series, inplace=True)
    return(gamma_df)


# compute coherence score (akin to LMD?)
def compute_coherence_values(dictionary, corpus, texts,start, limit, step):
    from gensim.models import CoherenceModel
    coherence_values = []
    model_list = []
    num_topics1 = [i for i in range(start, limit, step)]
    for num_topics in num_topics1:
        model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=100,update_every=1, chunksize=100, passes=10, alpha='auto', per_word_topics=True)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
    # note, list of 2 objs returned
    return model_list,coherence_values

## compute perplexity fit
def compute_perplexity_values(model_list, corpus,start, limit, step):
    
    perplexity_values = []
    for num_topics in range(start, limit, step):
        #model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=num_topics, random_state=100,
        #                                  update_every=1, chunksize=100, passes=10, alpha='auto', per_word_topics=True)
        model_index = num_topics - start
        model = model_list[model_index]
        perplexity_values.append(model.log_perplexity(corpus))
        #model_list.append(model)
        

    return perplexity_values  # note, list of 2 objs returned


def ltm_on_the_text_data(data13,col_names):
    
    start = time.time()
    
    for i in col_names:
        new_col_name = i+"_sentence"
        data14=data13[['date',new_col_name,'symbol']]

        data_des=data14.dropna()
        #converting dataframe to list
        description = data_des[new_col_name].to_list()
        
        corpus1 = textClean(description)
        corpus2 = [[word for word in ' '.join(doc).split()] for doc in corpus1]
        id2word1 = corpora.Dictionary(corpus2)  # Create dictionary
        corpus = [id2word1.doc2bow(text) for text in corpus2]
        
        a0 = [[(id2word1[id], freq) for id, freq in cp] for cp in corpus[:1]]
        # Build LDA model for (say) K=4 topics
        lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,id2word=id2word1,num_topics=5,random_state=100,
                                                   update_every=1,
                                                   chunksize=100,
                                                   passes=10,
                                                   alpha='auto',
                                                   per_word_topics=True)
        
        # invoke func
        beta_df = build_beta_df(lda_model=lda_model, id2word=id2word1)
        
        # invoke func
        gamma_df = build_gamma_df(lda_model=lda_model, corpus0=description,id2word=id2word1)
        
        row0 = gamma_df.values.tolist()
        row=[]
        for k in range(len(row0)):
            row1 = list(enumerate(row0[k]))
            row1_y = [y for (x,y) in row1]
            max_propn = sorted(row1_y, reverse=True)[0]
            row2 = [(k, x, y) for (x, y) in row1 if y==max_propn]
            row.append(row2)
        
        start1=2
        limit1=19
        step1=1
        
        model_list, coherence_values = compute_coherence_values(dictionary=id2word1, corpus=corpus,texts=corpus2,start=start1, limit=limit1, step=step1)
        
        coher = list(enumerate(coherence_values))  # create an index for each list elem
        index_max = [x for (x,y) in coher if y==max(coherence_values)]  # obtain index num corres to max coherence value
        Optimal_numTopics = int(str(index_max[0]))+2  # convert that list elem into integer (int()) via string (str())
        
        # perplexity_values = compute_perplexity_values(dictionary=id2word, corpus=corpus, start=2, limit=15, step=1)
        
        perplexity_values = compute_perplexity_values(model_list, corpus=corpus, start=start1, limit=limit1, step=step1)
        
        # compute optimal num_topics using perplexity based fit
        perpl = list(enumerate(perplexity_values))  # create an index for each list elem
        index_min = [x for (x,y) in perpl if y==min(perplexity_values)]  # obtain index num corres to max coherence value
        
        optimal_numTopics = int(str(index_min[0]))+2  # convert that list elem into integer (int()) via string (str())
        
        if Optimal_numTopics >= len(model_list):
            optimal_model = model_list[Optimal_numTopics-2]
        else:
            optimal_model = model_list[Optimal_numTopics]
            
        model_topics = optimal_model.show_topics(formatted=False)
        
        # Get main topic in each document
        gamma_df = build_gamma_df(lda_model=optimal_model, corpus0=description,id2word=id2word1)
        
        row0 = gamma_df.values.tolist()
        row=[]
        for l in range(len(row0)):
            row1 = list(enumerate(row0[l]))
            row1_y = [y for (x,y) in row1]
            max_propn = sorted(row1_y, reverse=True)[0]
            row2 = [(l, x, y) for (x, y) in row1 if y==max_propn]
            row.append(row2)
        
        
        sent_topics_df = pd.DataFrame()
        for row1 in row:
            for (doc_num, topic_num, prop_topic) in row1:
                wp = optimal_model.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(doc_num), int(topic_num), 
                                                                  round(prop_topic,4), 
                                                                  topic_keywords]), 
                                                               ignore_index=True)
        
        sent_topics_df.columns = [i+'_Doc_num', i+'_Dominant_Topic', i+'_Perc_Contribution', i+'_Topic_Keywords']

        # Add original text to the end of the output
        contents = pd.Series(description)
        sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
        #return(sent_topics_df)
        sent_topics_df.columns = [i+'_Doc_num', i+'_Dominant_Topic', i+'_Perc_Contribution', i+'_Topic_Keywords', i+'_contents']
        
        date=data_des[['date','symbol']]
        date=date.reset_index()
        date=date.drop(['index'],axis=1)

        sent_topics_df2 = pd.concat([sent_topics_df,date],axis=1)
        sent_topics_df3=sent_topics_df2[[i+'_Dominant_Topic',i+'_Perc_Contribution','date','symbol']]
        data13=pd.merge(data13,sent_topics_df3,left_on=["date","symbol"],right_on=["date","symbol"],how="left")
        #data13.to_csv('data/data_with_ltm'+i+'.csv')
    
    data13.to_csv('data/data_with_ltm.csv')
    
    print("Time Taken (sec): ",(time.time())-start)
    
    return data13

def get_predictions(s1,s2):
    sdate=date.today() - timedelta(days=5)
    sdate1 = sdate.strftime("%Y-%m-%d")
    edate = date.today()
    edate1  = edate.strftime("%Y-%m-%d")
    #l1 = get_stock_prices_ohlc_modified(s1,sdate,edate)
    #l1.to_csv('data/latest_data_reliance.csv')
    l1 = pd.read_csv('data/latest_data_reliance.csv')
    l1['Date'] = pd.to_datetime(l1['Date']).dt.date
    
    st.write("Getting Latest Stock Data")
    st.write("Getting Latest News Data")
    df = get_news_data_from_stock_edge(s1,2)
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    
    st.write("Getting Latest Twitter Data")
    df2 = get_data_from_twitter_upd(s1,n=2,sdate=sdate1,edate=edate1)
    df2['Date'] = pd.to_datetime(df2['Date']).dt.date
    
    df_with_news = l1.merge(df,on='Date',how='left')
    df_final = df_with_news.merge(df2,on='Date',how='left')
    
    col_names = ['text','twitter_text']
    
    st.write("Cleaning Data")
    cleaned_data = clean_the_data(df_final,col_names)
    st.write("Sentiment Analysis")
    df_with_sentiments = sentiment_analysis(cleaned_data,col_names)
    data12 = calculate_stock_indicators(df_with_sentiments,'close_14_ema','close_14_ema')
    st.write("LTM on text data")
    data12 = data12.fillna(method='ffill')
    data12 = data12.tail(1)
    #df_final_1 = ltm_on_the_text_data(data12,col_names)
    #df_final_1.to_csv('data/predict.csv')
    df_final_1 = pd.read_csv('data/predict.csv')
    
    st.write("Predicting the values")
    loaded_model = pickle.load(open('model/l1.pkl', 'rb'))
    print(loaded_model)
    df_final_1 = df_final_1.tail(1)
    df_final_12 = df_final_1[['volume','text_compound','twitter_text_compound','text_Perc_Contribution','twitter_text_Perc_Contribution','close_14_ema_x']]
    df_final_12 = df_final_12.fillna(0)
    result = loaded_model.predict(df_final_12)
    df_1 = pd.DataFrame(result,columns=['Predicted Close'])
    df_1['Actual Close'] = df_final_1['close'].iloc[0]
    df_1['Date'] = df_final_1['date'].iloc[0]
    df_1['Company'] = s1
    st.write(df_1)

#get_predictions('RELIANCE','S1')
    
