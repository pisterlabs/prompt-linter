# %%
import requests
import finnhub
import warnings
import time
import numpy as np
import pandas as pd
from yahoo_fin import stock_info
from bs4 import BeautifulSoup
from datetime import datetime
from datetime import date,timedelta
from yahoo_fin import stock_info
import re
import nltk
import operator
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.stem import WordNetLemmatizer 
from gensim.corpora import Dictionary
from gensim import models
from gensim.models import CoherenceModel
from gensim.models.nmf import Nmf
from gensim.models import LdaModel,LdaMulticore
from functools import reduce
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

# %%
import pickle

# %%
warnings.filterwarnings('ignore')

# %%
#ticker = 'AAPL'
#company_name = 'Apple'

# %%
finnhub_client = finnhub.Client(api_key = "c7i4q2qad3if83qgdfl0")

# %%
def get_company_news(ticker):
    news = []
    initial_date = datetime.now().date()
    offset = (pd.to_datetime(initial_date) - pd.DateOffset(days = 1)).strftime('%Y-%m-%d')
    news_iter = finnhub_client.company_news(ticker, _from = offset, to = initial_date)
    news += news_iter

    #initial_date = (pd.to_datetime(offset) - pd.DateOffset(days = 1)).strftime('%Y-%m-%d')

    return news



# %%
#checking the correctness


# %%
# в 19 строчке надо определить время, после которого отсекать
def get_news_dataset(ticker, company_name, data):
    #extracting news from dictionary 
    news = pd.DataFrame(data[ticker])
    news['datetime'] = news['datetime'].apply(lambda date: datetime.utcfromtimestamp(date).strftime('%Y-%m-%d %H:%M:%S'))

    #only news with ticker occurance
    news = news[news.headline.str.contains(company_name)].reset_index(drop = True)

    #GMT-to-EST correction
    news['datetime'] = news['datetime'].apply(lambda date: pd.to_datetime(date) - pd.DateOffset(hours = 5))
    
    #next day correction
    timestamps = news['datetime'] 
    timestamps_new = []
    for date in timestamps:
        date = date.strftime('%Y-%m-%d %H:%M:%S')
        if pd.to_datetime(date) > pd.to_datetime(f'{date[:10]} 10:00:00'):
            date = pd.to_datetime(date)
            date_new = (date - timedelta(hours = date.hour, minutes = date.minute,
                    seconds = date.second)) + timedelta(days = 1)
            timestamps_new.append(date_new.strftime('%Y-%m-%d %H:%M:%S'))
        else:
            timestamps_new.append(date)


    news['datetime'] = timestamps_new
    
    news = news[['datetime', 'headline', 'source', 'summary']]
    news['datetime'] = news['datetime'].apply(lambda date: pd.to_datetime(date).strftime('%Y-%m-%d'))
    
    #trading days only
    bdays = pd.bdate_range(start = news['datetime'].to_list()[-1], 
               end = news['datetime'][0])

    bdays = pd.DataFrame(bdays, columns = ['datetime'])

    bdays.datetime = bdays.datetime.astype('str')

    news_filtered = pd.merge(news, bdays, how = 'right')
    
    #dropping dublicates
    news_filtered.drop_duplicates(['headline'], ignore_index = True, inplace = True)
    
    return news_filtered



# %% [markdown]
# # Financials

# %% [markdown]
# #### Relative Strength Index

# %%
#здесь выбрать какое среднее брать - обычное или эксп
def rsi(historical_data, window):
    changes = historical_data['adjclose'].diff(periods = 1)
    gain, loss = changes.clip(lower = 0), changes.clip(upper = 0).abs()
    
    #rsi based on SMA
    rolling_gain = gain.rolling(window, closed = 'left').mean()
    rolling_loss = loss.rolling(window, closed = 'left').mean()
    
    rs = rolling_gain / rolling_loss
    rsi_sma = 100.0 - (100.0 / (1.0 + rs))
    
    #rsi based on EMA
    rolling_gain_exp = rolling_gain.ewm(span = window).mean()
    rolling_loss_exp = rolling_loss.ewm(span = window).mean()

    rs_exp = rolling_gain_exp / rolling_loss_exp
    rsi_ema = 100.0 - (100.0 / (1.0 + rs_exp))
    
    return rsi_sma, rsi_ema

# %% [markdown]
# #### Financial indicators for each day

# %%
def financial_indicators(historical_data):
    df = {}
    data = historical_data['adjclose']
    df['datetime'] = historical_data['adjclose'].index
    df['adjclose'] = data
    #10 days Moving Average
    df['ma_10'] = data.rolling(window = 10, closed = 'left').mean()[10:]
    #20 days Moving Average
    df['ma_20'] = data.rolling(window = 20, closed = 'left').mean()[20:]
    #10 days Moving Average
    df['ma_30'] = data.rolling(window = 30, closed = 'left').mean()[30:]
    
    #12 days Exponential Moving Average
    df['ema_12'] = data.ewm(span = 12).mean()[1:]
    df['ema_12'] =  df['ema_12'].shift()
    #26 days Exponential Moving Average
    df['ema_26'] = data.ewm(span = 26).mean()[1:]
    df['ema_26'] =  df['ema_26'].shift()
    #6 days Relative Strength Index
    df['rsi_6_sma'], df['rsi_6_ema'] = rsi(historical_data, window = 6)
    #12 days Relative Strength Index
    df['rsi_12_sma'], df['rsi_12_ema'] = rsi(historical_data, window = 12)
    #24 days Relative Strength Index
    df['rsi_24_sma'], df['rsi_24_ema'] = rsi(historical_data, window = 24)
    
    final_df = pd.DataFrame(df)
   
    return final_df

# %%
def get_financials_dataset(ticker):
    #historical data
    historical_data = stock_info.get_data(ticker,
                                      start_date = pd.to_datetime(news_filtered['datetime'][0]) - 
                                      pd.DateOffset(days = 100), 
                                      end_date =  news_filtered['datetime'].to_list()[-1],
                                      index_as_date = True, 
                                      interval = '1d')
    
    #financial indicators
    financials = financial_indicators(historical_data)
    financials = financials[financials['datetime'] >= pd.to_datetime(news_filtered.datetime[0])]
    financials['datetime'] = financials['datetime'].apply(lambda date: date.strftime('%Y-%m-%d'))
    
    #reseting the index for further concatenation with news dataset
    financials.index = financials.index.astype('object')
    fin_index = pd.Series(financials.index)
    fin_index = fin_index.apply(lambda fin : fin.strftime('%Y-%m-%d'))
    financials.set_index(fin_index, drop = True, inplace = True)
    
    return financials

# %% [markdown]
# ### Reshaping the data to long format

# %%
def reshape_news_data(news_filtered):
    news_filtered['RANK'] = news_filtered.groupby("datetime")["datetime"].rank(method="first", ascending=True)
    news_filtered['RANK'] = news_filtered['RANK'].astype('int')
    news_reshaped = news_filtered.pivot(index = 'datetime', values = 'headline', columns = 'RANK')
    news_reshaped.columns = [f'news_{index}' for index in news_reshaped.columns]
    
    return news_reshaped

# %%
def get_single_prediction_data(ticker, company_name, data ):

    #news 
    news_filtered = get_news_dataset(ticker, company_name, data)

    #historical prices
    historical_data = stock_info.get_data(ticker,
                                        start_date = pd.to_datetime(news_filtered['datetime'][0]) - 
                                        pd.DateOffset(days = 100), 
                                        end_date =  news_filtered['datetime'].to_list()[-1],
                                        index_as_date = False, 
                                        interval = '1d')
    historical_data.drop_duplicates('date', inplace = True)
    financials = financial_indicators(historical_data).iloc[-1, :]
    #news reshaped
    news_reshaped = reshape_news_data(news_filtered)
    news_reshaped = news_reshaped.iloc[0, :]
    financials.name = news_reshaped.name
    final_dataframe = pd.concat([news_reshaped, financials], axis = 0, join = 'inner')
    index = news_reshaped.index.append(financials.index)
    values = news_reshaped.values.tolist() + financials.values.tolist() 
    df = pd.DataFrame(values).transpose()
    df.columns = index
    df.drop('datetime', inplace = True, axis = 1)
    df.drop(columns = ['rsi_12_ema', 'rsi_24_ema'], axis = 1, inplace = True)
    df.reset_index(inplace=True)
    df['index'] = datetime.today().strftime('%Y-%m-%d')
    df.set_index('index', inplace = True)
    return df

# %%
def get_target(historical_data, predict_trend = True):
    historical_data['diff'] = historical_data['adjclose'] - historical_data['open']
    movements = historical_data['diff'].apply(lambda price: 1 if price > 0 else 0)
    target = pd.DataFrame({'date':historical_data['date'], 'target' : movements}).set_index('date')
    
    target.index = target.index.astype('object')

    target_index = pd.Series(target.index)
    target_index = target_index.apply(lambda fin : fin.strftime('%Y-%m-%d'))
    target.set_index(target_index, drop = True, inplace = True)
    if predict_trend:
        #trend
        trend = []
        i = 0
        i_initial = 0
        for i in range(len(target) - 1):
            try:
                if target.target[i]:
                    while target.target[i] == 1:
                        i += 1

                else:
                    while target.target[i] == 0:
                        i += 1

                trend.append(i - i_initial)

                i_initial += 1
                i = 0
            except:
                break
        #appending trend
        target['trend'] = np.NaN
        index = len(target) - len(trend)
        target['trend'][:-index] = trend
        target = target.iloc[:-index, :]

        target.trend = target.trend.astype('int')

        return target
    else:
        return target




# ### Sentiment and Topics


# %%
#cleaning
def clean_data_single_prediction(df):
    df.dropna(axis=1, inplace = True)
    news = df[[col for col in df.columns if col.startswith('news')]]
    news = news.values.tolist()[0]
    news_cleaned = []
    news_per_day = []
    for text in news:
        text = re.sub('[^a-zA-Z0-9]+\s*', ' ', text) #not a number or a letter
        text = text.lower() #lowercase
        news_per_day.append(text)

    news_cleaned.append(news_per_day)

    lemmatizer = WordNetLemmatizer()
    news_lemmatized = []

    news_per_day = []
    for text in news_cleaned[0]:
        text = [lemmatizer.lemmatize(word) for word in nltk.word_tokenize(text)]
        text = [''.join(lemma) for lemma in text]
        text = ' '.join(text)
        news_per_day.append(text)

    news_lemmatized.append(news_per_day)

    stopwords_ = stopwords.words('english')
    news_cleaned = []
    news_per_day = []
    for text in news_lemmatized[0]:
        text = [word for word in text.split(' ') if word not in stopwords_]
        text = ' '.join(text)
        news_per_day.append(text)

    news_cleaned.append(news_per_day)

    return news_cleaned

# %%
def get_sentiment_lm(long_news):
    lm = pd.read_csv('dictionaries/lm.csv')
    lm_sentiment = pd.merge(long_news, lm, how = 'left').set_index(long_news.index)
    lm_sentiment = pd.DataFrame(lm_sentiment.groupby(['day', 'ticker', 'news'])['binary_score'].mean())
    lm_sentiment = lm_sentiment.reset_index().pivot(index = ['day', 'ticker'], 
                                                values = 'binary_score', columns = 'news')
    lm_sentiment.columns = [f'sentiment_{index}' for index in lm_sentiment.columns]
    
    return lm_sentiment

# %%
def get_sentiment_oliveira(long_news):
    oliveira = pd.read_csv('dictionaries/oliveira.csv')
    ol_sentiment = pd.merge(long_news, oliveira, how = 'left').set_index(long_news.index)
    ol_sentiment = pd.DataFrame(ol_sentiment.groupby(['day', 'ticker', 'news'])['score'].mean())
    ol_sentiment = ol_sentiment.reset_index().pivot(index = ['day', 'ticker'], 
                                                values = 'score', columns = 'news')
    ol_sentiment.columns = [f'sentiment_{index}' for index in ol_sentiment.columns]
    
    return ol_sentiment

# %%
def get_sentiment_sentic(long_news):
    sentic = pd.read_csv('dictionaries/sentic.csv')
    sen_sentiment = pd.merge(long_news, sentic, 
                         how = 'left')[['word', 'polarity_intensity']].set_index(long_news.index)
    sen_sentiment = pd.DataFrame(sen_sentiment.groupby(['day', 'ticker', 'news'])['polarity_intensity'].mean())
    sen_sentiment = sen_sentiment.reset_index().pivot(index = ['day', 'ticker'], 
                                                values = 'polarity_intensity', columns = 'news')
    sen_sentiment.columns = [f'sentiment_{index}' for index in sen_sentiment.columns]
    
    return sen_sentiment

# %%
def imputer(sentiment_table, min_news = 0, max_news = 40):
    sentiment_table = sentiment_table.dropna(axis = 0, thresh = min_news).iloc[:, : max_news]
    impute_values = sentiment_table.mean(axis = 1)
    for i in range(len(sentiment_table)):
        sentiment_table.iloc[i, :].fillna(impute_values[i], inplace = True)
        
    return sentiment_table

# %%
def scale_sentiment(sentiment_df, scaler):
    scaled = pd.DataFrame(scaler.transform(imputer(sentiment_df)),
                          columns = imputer(sentiment_df).columns,
                          index = imputer(sentiment_df).index)
    
    return scaled



# %%
def intersection_index(arr1, arr2, arr3):
    # Converting the arrays into sets
    s1 = set(arr1)
    s2 = set(arr2)
    s3 = set(arr3)
      
    # Calculates intersection of 
    # sets on s1 and s2
    set1 = s1.intersection(s2)
      
    # Calculates intersection of sets
    # on set1 and s3
    result_set = set1.intersection(s3)
      
    # Converts resulting set to list
    final_list = list(result_set)
    return final_list


def combine_sentiments(sentiment1, sentiment2, sentiment3):
    #common indices
    indices = intersection_index(sentiment1.index, sentiment2.index, sentiment3.index)
    indices = pd.MultiIndex.from_tuples(indices, names = ["day", "ticker"])
    #empty frame
    final_sentiment = pd.DataFrame(index = indices, columns = sentiment1.columns)
    
    #fulfilling the frame
    final_sentiment = pd.DataFrame(index = indices, columns = sentiment1.columns)
    for index in indices:
        for column in sentiment1.columns:
            sent_value = 0.4 * sentiment1.loc[index, column] +  0.2 * sentiment2.loc[index, column] + 0.4 * sentiment3.loc[index, column]
            
            final_sentiment.loc[index, column] = sent_value
            
    return final_sentiment




# %%
#model
#model = pickle.load(open('binary_model.sav', 'rb'))
#threshold = 0.36448651000920534

# %%
#model.predict_proba(final_data_for_prediction)[::, 1] > threshold

# %%
def get_all_data(ticker, company_name):
    #get news
    data = {}
    tickers = [ticker]
    for ticker in tickers:
        start = time.time()
        data[ticker] = get_company_news(ticker)
        end = time.time() - start
        print(f'{ticker} news is parsed!')
        #print(f'Sleeping for {np.round((62 - end), 0)} seconds...')
        #time.sleep((60 - end))
        if ticker != tickers[-1]:
            time.sleep(70)
        else:
            continue
    
    #historical data
    news_filtered = get_news_dataset(ticker, company_name, data)
    historical_data = stock_info.get_data(ticker,
                                        start_date = pd.to_datetime(news_filtered['datetime'][0]) - 
                                        pd.DateOffset(days = 100), 
                                        end_date =  news_filtered['datetime'].to_list()[-1],
                                        index_as_date = False, 
                                        interval = '1d')
    #news + financials
    df = get_single_prediction_data(ticker, company_name, data)

    #sentiment data
    #news = df[[col for col in df.columns if col.startswith('news')]]
    #news = news.values.tolist()[0]

    #cleaned news
    news_cleaned = clean_data_single_prediction(df)

    #long news
    day_index = []
    news_index = []
    word_index = []
    corpus = []
    tickers_index = []
    dates = df.reset_index()['index']
    for i, day in enumerate(news_cleaned): #news_cleaned
        for j, news in enumerate(day):
            for k, text in enumerate(news.split(' ')):
                day_index.append(dates[0])
                tickers_index.append(ticker)
                news_index.append(j)
                word_index.append(k)
                corpus.append(news.split(' ')[k])

    tuples = list(zip(day_index, tickers_index, news_index, word_index))
    multindex = pd.MultiIndex.from_tuples(tuples, names = ["day", "ticker", "news", "word_count"])
    long_news = pd.DataFrame({'word': corpus}, index = multindex)
    

    #getting sentiments
    scaler_lm = pickle.load(open('scalers/scaler_lm.sav', 'rb'))
    scaler_ol = pickle.load(open('scalers/scaler_ol.sav', 'rb'))
    scaler_sen = pickle.load(open('scalers/scaler_sen.sav', 'rb'))
    ol_sentiment = get_sentiment_oliveira(long_news)
    lm_sentiment = get_sentiment_lm(long_news)
    sen_sentiment = get_sentiment_sentic(long_news)
    for i in range(ol_sentiment.shape[1], 41):
        ol_sentiment[f'sentiment_{i}'] = np.NaN

    for i in range(lm_sentiment.shape[1], 41):
        lm_sentiment[f'sentiment_{i}'] = np.NaN

    for i in range(sen_sentiment.shape[1], 41):
        sen_sentiment[f'sentiment_{i}'] = np.NaN

    #scaling sentiments
    ol_scaled = imputer(ol_sentiment, min_news = 1, max_news = 40)
    ol_scaled = scale_sentiment(ol_scaled, scaler_ol)

    lm_scaled = imputer(lm_sentiment, min_news = 1, max_news = 40)
    lm_scaled = scale_sentiment(lm_scaled, scaler_lm)

    sen_scaled = imputer(sen_sentiment, min_news = 1, max_news = 40)
    sen_scaled = scale_sentiment(sen_scaled, scaler_sen)

    #combining sentiments
    final_sentiment = 0.4 * ol_scaled + 0.2 * lm_scaled  + 0.4 * sen_scaled

    #topic modelling
    nmf = Nmf.load('nmf')
    dictionary = Dictionary.load('dictionary_nmf')
    other_corpus = [dictionary.doc2bow(text.split()) for text in news_cleaned[0]]
    topics = []
    for i in range(len(other_corpus)):
        topics.append(nmf.get_document_topics(other_corpus[i]))
    topics = reduce(lambda x, y: x + y, topics)
    topics_sorted = sorted(topics, key = lambda x:x[1], reverse = True)[:5]
    topics_final = [x[0] for x in topics_sorted]
    topics_final = pd.DataFrame(topics_final, index = [f'topic_{i}' for i in range(5)]).transpose()

    #final data
    financials = df[[col for col in df.columns if 'news' not in col]]
    financials['target_3_days_previous'] = get_target(historical_data, predict_trend=False).target.rolling(3, closed = 'left').mean()[-1]
    financials['target_5_days_previous'] = get_target(historical_data, predict_trend=False).target.rolling(5, closed = 'left').mean()[-1]
    final_data_for_prediction = pd.concat([financials.reset_index(drop = True), final_sentiment.reset_index(drop = True), 
    topics_final.reset_index(drop = True)], axis = 1)

    return final_data_for_prediction

# %%
#get_all_data('AAPL', 'Apple')



# %%
