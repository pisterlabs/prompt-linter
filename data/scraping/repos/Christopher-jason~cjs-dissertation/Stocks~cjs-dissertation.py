# %%
'''
Project: Impact of r/wallstreetbets analysed using LSTM and CNN

Date: 26/8/2023

Author: Christopher Jason Sagayaraj

Sentiment analysis to understand the effect of meme community on the stock market. 
This paper studies how r/wallstreetbets have an impact on the stock market and how some companies are 
discussed more due to the relevance/popularity of the company in the general space.

'''
# %%
# Importing Libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import yfinance as yf
import openai
from os import getenv
from dotenv import load_dotenv
import openai
import datetime
import time
import seaborn as sns

# %%
####### GET THE TOP 13 SELECTED STOCKS #########

symbols = pd.read_csv('symbols/symbols_top_20.csv')
symbols.head(3)

# %%
######### REMOVE DUPLICATES ####### 

def remove_duplicates(input_file, output_file):
    try:
        with open(input_file, 'r') as f:
            sentences = f.readlines()

        # Remove duplicates while preserving the order
        unique_sentences = list(dict.fromkeys(sentences))

        with open(output_file, 'w') as f:
            f.writelines(unique_sentences)

        print("Duplicates removed successfully and saved to", output_file)
    except FileNotFoundError:
        print("File not found. Please check the input file path.")
    except Exception as e:
        print("An error occurred:", e)

# %%
# hot files
input_file = 'raw_txt/hot.txt'
output_file = 'clean_txt/clean_hot.txt'

remove_duplicates(input_file, output_file)

# New file
input_file = 'raw_txt/new.txt'
output_file = 'clean_txt/clean_new.txt'

remove_duplicates(input_file, output_file)

# %%
######### CONVERT THE LINES TO A DATAFRAME #############
def make_dataframe(file_name):
    rows = []

    with open(file_name, 'r') as file:
        current_date = None
        for line in file:
            # Check if the line matches the date format
            match = re.search(r'[-]+(\d{4}-\d{2}-\d{2})', line)
            if match:
                # Extract the date in the format "YYYY-MM-DD"
                current_date = match.group(1) # Only extract the date part without dashes
            else:
                if current_date is not None: # Avoid adding rows without an associated date
                    rows.append({'date': current_date, 'text': line.strip()})

    df = pd.DataFrame(rows, columns=['date', 'text'])
    return df

# %%
hot_df = make_dataframe('clean_txt/clean_hot.txt')
new_df = make_dataframe('clean_txt/clean_new.txt')

# %%
#write variables into a csv file 
hot_df.to_csv('text_df/hot_df.csv', index=False)
new_df.to_csv('text_df/new_df.csv', index=False)

#read the df into the variable
hot_df = pd.read_csv('text_df/hot_df.csv')
new_df = pd.read_csv('text_df/new_df.csv')

# %%
######### FIND TICKERS IN THE TEXT ##########

def symbol_matches(sentence, symbol_list):
    matches = []
    for symbol in symbol_list:
        if pd.notna(symbol): 
            pattern = r"\b" + re.escape(str(symbol)) + r"\b"
            if re.search(pattern, sentence):
                matches.append(symbol)
    return matches

def find_matches(hot_df, symbols):
    results = []
    for index, row in hot_df.iterrows():
        date = row['date']
        sentence = row['text']
        try:
            # Find matches for each sentence with "Symbol" and "Dollar_Symbol" columns
            matches_symbols = symbol_matches(sentence, symbols['Symbol'].tolist())
            matches_dollar_symbols = symbol_matches(sentence, symbols['Dollar_Symbol'].tolist())
            all_matches = matches_symbols + matches_dollar_symbols
        except Exception as e:
            print(e)
            print(sentence)
            break;
        # If no match is found then input "S&P"
        if not all_matches:
            all_matches = ['S&P']

        ticker_string = ', '.join(all_matches)
        results.append((date, sentence, ticker_string))
    tickers = pd.DataFrame(results, columns=["date", "text", "ticker"])
    return tickers

# %%
hot_ticker = find_matches(hot_df,symbols)
new_ticker = find_matches(new_df,symbols)

# %%
#write variables to a csv with no index
hot_ticker.to_csv('text_df/hot_ticker.csv', index=False)
new_ticker.to_csv('text_df/new_ticker.csv', index=False)

#read the csv files
hot_ticker = pd.read_csv('text_df/hot_ticker.csv')
new_ticker = pd.read_csv('text_df/new_ticker.csv')

# %%
##### FINBERT #####

# Use a pipeline as a high-level helper
from transformers import pipeline
pipe_finbert = pipeline("text-classification", model="ProsusAI/finbert")

def finbert_sentiments(df,pipe=pipe_finbert):
    rows = []
    for index, row in df.iterrows():
        date = row['date']
        text = row['text']
        ticker = row['ticker']

        if isinstance(text, str):
            sentences = [text]
        #send to pipeline
        results = pipe(sentences)

        for sentence, result in zip(sentences, results):
            rows.append((date, sentence, ticker, result['label'], result['score']))

    finbert_sentiment = pd.DataFrame(rows, columns=['date', 'text', 'ticker', 'label', 'score'])
    
    return finbert_sentiment

hot_finbert = finbert_sentiments(hot_ticker['text'].str.lower())
new_finbert = finbert_sentiments(new_ticker['text'].str.lower())

# %%
#write variable into a csv with no index
hot_finbert.to_csv('sentiment_scores/hot_finbert.csv', index=False)
new_finbert.to_csv('sentiment_scores/new_finbert.csv', index=False)

# %%
####### VADER #######

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer_vader = SentimentIntensityAnalyzer()

def vader_sentiments(df,analyzer=analyzer_vader):
    rows = []
    
    for index, row in df.iterrows():
        date = row['date']
        text = row['text']
        ticker = row['ticker']
        if isinstance(text, str):
            sentences = [text]
        #send to pipeline
        for sentence in sentences:
            result = analyzer.polarity_scores(sentence)
            rows.append((date, sentence, ticker, result['neg'], result['neu'], result['pos'], result['compound']))

    vader_sentiment = pd.DataFrame(rows, columns=['date', 'text', 'ticker', 'neg', 'neu', 'pos', 'compound'])
    
    return vader_sentiment

hot_vader = vader_sentiments(hot_ticker)
new_vader = vader_sentiments(new_ticker)  
# %%
#write the variable to csv file with no index
hot_vader.to_csv('sentiment_scores/hot_vader.csv', index=False)
new_vader.to_csv('sentiment_scores/new_vader.csv', index=False)

# %%
########## ROBERTA #############

# Use a pipeline as a high-level helper
from transformers import pipeline
pipe_roberta = pipeline("text-classification", model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")

def roberta_sentiments(df,pipe=pipe_roberta):
    rows = []
    
    for index, row in df.iterrows():
        date = row['date']
        text = row['text']
        ticker = row['ticker']
        if isinstance(text, str):
            sentences = [text]
        results = pipe(sentences)
        
        for sentence, result in zip(sentences, results):
            rows.append((date, sentence, ticker, result['label'], result['score']))

    roberta_sentiment = pd.DataFrame(rows, columns=['date', 'text', 'ticker', 'label', 'score'])
    return roberta_sentiment


hot_roberta = roberta_sentiments(hot_ticker['text'].str.lower())
new_roberta = roberta_sentiments(new_ticker['text'].str.lower())

# %%
#write sentiments into a csv with no index
hot_roberta.to_csv('sentiment_scores/hot_roberta.csv', index=False)
new_roberta.to_csv('sentiment_scores/new_roberta.csv', index=False)

# %%
######## GPT-3 ########
load_dotenv()
openai.api_key = getenv('OPENAI_API_KEY')

tmp_df = pd.DataFrame()

# %%
def get_response(content):
    return openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "Analyze the given text and classify it into: negative, or positive. Also provide a sentiment score within the range of -1 to 1. Score values must be calculated with high precision with up to three decimal places. Your response format should be: sentiment, score e.g., ('negative, -0.145')."
            },
            {
                "role": "user",
                "content": content
            }
        ],
        temperature=1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
def chatgpt_df(rows):
    chatgpt = pd.DataFrame(rows, columns=['date', 'text', 'ticker', 'label', 'score'])
    return chatgpt

def chatgpt_sentiments(df, tmp_df=tmp_df, max_retries=5, base_delay=1, analyzer=openai):
    rows = []
    total_tokens = 0
    calls_per_minute = 1000
    call_interval = 60 / calls_per_minute
    last_call_time = 0
    last_token_reset_time = time.time() # Keep track of when the token count was last reset
    max_loop_count = 10 * max_retries # Maximum number of consecutive loop iterations without progress
    
    for index, row in df.iterrows():
        date = row['date']
        text = row['text']
        ticker = row['ticker']
        retries = 0
        loop_count = 0 # Counter for loop iterations without progress
        
        while retries < max_retries:
            current_time = time.time()
            time_since_last_call = current_time - last_call_time
            time_since_last_token_reset = current_time - last_token_reset_time

            if time_since_last_token_reset >= 60:
                total_tokens = 0 # Reset token count if it's been more than a minute
                last_token_reset_time = current_time

            if time_since_last_call < call_interval or total_tokens >= 85000:
                sleep_time = call_interval - time_since_last_call
                print(f"Rate limit reached, sleeping for {sleep_time} seconds...")
                time.sleep(sleep_time)
                loop_count += 1
                if loop_count > max_retries:  # Break out of the loop if too many iterations have occurred
                    print("Max loop count reached, unable to process the request.")
                    break
                continue 
                
                loop_count = 0
            try:
                last_call_time = time.time() # Update the time of the last successful call
                result = get_response(text)
                tokens = result['usage']['total_tokens']
                total_tokens += tokens # Update the total token count
                content = result.choices[0].message['content']
                try:
                    label, score_str = content.split(',', 1) # Split only at the first comma
                    label = label.strip()
                    score = float(score_str.strip())
                except ValueError:
                    print(f"Error here: {content}")
                    label = "error"
                    score = 2
                
                rows.append((date, text, ticker, label, score))
                print(f"Sentiment captured: {score}, token_used: {tokens}, total for the min: {total_tokens}")
                break

            except openai.error.ServiceUnavailableError:
                tmp_df = chatgpt_df(rows) #save the answers till the error point.
                tmp_df.to_csv('tmp.csv', index=False) #write tmp_df to a csv file named tmp with no index
                print("Did not get result.")
                delay = base_delay * (2 ** retries)
                print(f"Service unavailable, retrying in {delay} seconds...")
                time.sleep(delay)
                retries += 1
            except Exception as e:  # Catch-all for other exceptions
                tmp_df = chatgpt_df(rows) #save the answers till the error point.
                tmp_df.to_csv('tmp.csv', index=False)
                print(f"Unexpected error: {e}")
                delay = base_delay * (2 ** retries)
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
                retries += 1
            except KeyboardInterrupt:
                print("Operation was interrupted by the user.")
                tmp_df = chatgpt_df(rows) #save the answers till the error point.
                tmp_df.to_csv('tmp.csv', index=False)
                return chatgpt_df(rows)
        else: # This will execute if the while loop ends without a break statement
            print("Max retries reached, unable to process the request.")

    return chatgpt_df(rows)

# %%
hot_chatgpt = chatgpt_sentiments(hot_ticker)
new_chatgpt = chatgpt_sentiments(new_ticker)

#retry the rows which have an error
hot_tmp = chatgpt_sentiments(hot_chatgpt[hot_chatgpt['score']==2])
# Iterate through the merged DataFrame and update the label and score in chatgpt
hot_merged_df = pd.merge(hot_chatgpt, hot_tmp, on='text', suffixes=('_hot', '_tmp'))
for index, row in hot_merged_df.iterrows():
    hot_index = hot_chatgpt[hot_chatgpt['text'] == row['text']].index.item()
    hot_chatgpt.at[hot_index, 'label'] = row['label_tmp']
    hot_chatgpt.at[hot_index, 'score'] = row['score_tmp']

new_tmp = chatgpt_sentiments(new_chatgpt[new_chatgpt['score']==2])
new_merged_df = pd.merge(new_chatgpt, new_tmp, on='text', suffixes=('_new', '_tmp'))
for index, row in new_merged_df.iterrows():
    new_index = new_chatgpt[new_chatgpt['text'] == row['text']].index.item()
    new_chatgpt.at[new_index, 'label'] = row['label_tmp']
    new_chatgpt.at[new_index, 'score'] = row['score_tmp']

# %%
#clean the label 
hot_chatgpt['label'] = hot_chatgpt['label'].replace({
    'Positive': 'positive',
    'Neutral': 'neutral',
    'Negative': 'negative',
    'No sentiment': 'neutral',
    'neither': 'neutral'
})

new_chatgpt['label'] = new_chatgpt['label'].replace({
    'Positive': 'positive',
    'Neutral': 'neutral',
    'Negative': 'negative',
    'No sentiment': 'neutral',
    'neither': 'neutral'
})

#remove the rows from the df where the label is 'error'
hot_chatgpt = hot_chatgpt[hot_chatgpt['label'] != 'error']
new_chatgpt = new_chatgpt[new_chatgpt['label'] != 'error']

#remove the rows from the df where the label is 'error'
hot_chatgpt = hot_chatgpt[hot_chatgpt['label'] != 'error']
new_chatgpt = new_chatgpt[new_chatgpt['label'] != 'error']

# write the df to csv file with no index
hot_chatgpt.to_csv('sentiment_scores/hot_chatgpt.csv', index=False)
new_chatgpt.to_csv('sentiment_scores/new_chatgpt.csv', index=False)

# %%
#read the hot_chatgpt variable
hot_chatgpt = pd.read_csv('sentiment_scores/hot_chatgpt.csv')
new_chatgpt = pd.read_csv('sentiment_scores/new_chatgpt.csv')

# %%
######## DATA ANALYSIS #########

hot_finbert = pd.read_csv('sentiment_scores/hot_finbert.csv')
hot_vader = pd.read_csv('sentiment_scores/hot_vader.csv')
hot_chatgpt = pd.read_csv('sentiment_scores/hot_chatgpt.csv')
hot_roberta = pd.read_csv('sentiment_scores/hot_roberta.csv')
new_finbert = pd.read_csv('sentiment_scores/new_finbert.csv')
new_vader = pd.read_csv('sentiment_scores/new_vader.csv')
new_chatgpt = pd.read_csv('sentiment_scores/new_chatgpt.csv')
new_roberta = pd.read_csv('sentiment_scores/new_roberta.csv')


# %%
from sklearn.feature_extraction.text import CountVectorizer

def correlation_matrix(hot_sentiment, new_sentiment):

    sentiment_data = pd.concat([hot_sentiment, new_sentiment], ignore_index=True)
    sentiment_data.drop_duplicates(subset=['text'], inplace=True)
    # Using CountVectorizer to create a word matrix and removing stop words
    vectorizer = CountVectorizer(stop_words='english', max_features=10)
    word_matrix_vectorized = vectorizer.fit_transform(sentiment_data['text'])

    # Calculating the correlation matrix using CountVectorizer
    correlation_matrix_vectorized = np.corrcoef(word_matrix_vectorized.toarray(), rowvar=False)

    # Plotting the correlation matrix using CountVectorizer
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix_vectorized, annot=True, cmap='coolwarm', xticklabels=vectorizer.get_feature_names_out(), yticklabels=vectorizer.get_feature_names_out())
    plt.title('Correlation Matrix of Top 10 Words')
    plt.show()

# Words are same in all models, so running it once is good
correlation_matrix(hot_finbert, new_finbert)

# %%
def model_analysis(hot_sentiment, new_sentiment, model):
    print(f" ========= Analysis of the sentiment scores derived from {model} ========= \n\n")
    sentiment_data = pd.concat([hot_sentiment, new_sentiment], ignore_index=True)
    sentiment_data.drop_duplicates(subset=['text'], inplace=True)

    sentiment_counts = sentiment_data['label'].value_counts(normalize=True) * 100

    color_map = {
        'neutral': 'lightblue',
        'negative': 'red',
        'positive': 'green'
    }    
    # Plotting the overall sentiment distribution
    pie_colors = [color_map[label] for label in sentiment_counts.index]
    plt.figure(figsize=(5, 5))
    sentiment_counts.plot.pie(autopct='%1.1f%%',colors=pie_colors)
    plt.title(f'Overall Sentiment Distribution - {model}')
    plt.show()

    # Accesing the top 5 tickers labels
    top_6 = sentiment_data['ticker'].value_counts().head(6)
    top_5_no_sp = top_6[1:6]

    top_5_ticker_analysis = {}
    for ticker in top_5_no_sp.index:
        top_5_ticker_analysis[ticker] = (sentiment_data[sentiment_data['ticker'] == ticker]['label'].value_counts(normalize=True) * 100).to_dict()

    top_5_ticker_df = pd.DataFrame.from_dict(top_5_ticker_analysis, orient='index').fillna(0)
    print(top_5_ticker_df.head())

    # Plotting the sentiment distribution for the top 5 tickers
    bar_colors = [color_map[label] for label in top_5_ticker_df.columns]
    top_5_ticker_df.plot(kind='bar', stacked=True, figsize=(10, 6), color=bar_colors)
    plt.title(f'Sentiment Distribution for Top 5 Tickers - {model} ')
    plt.xlabel('Ticker')
    plt.ylabel('Percentage')
    plt.xticks(rotation=45)
    plt.show()

    #Plotting the distribution of the sentiment scores
    plt.figure(figsize=(10, 6))
    sns.histplot(sentiment_data['score'], kde=True, bins=20, color='purple')
    plt.title(f'Distribution of Sentiment Scores - {model}')
    plt.xlabel('Sentiment Score')
    plt.ylabel('Frequency')
    plt.show()

    
model_analysis(hot_finbert, new_finbert, 'Finbert')
model_analysis(hot_roberta, new_roberta, 'RoBERTa')
model_analysis(hot_chatgpt, new_chatgpt, 'GPT-3')

# %%
def model_analysis_vader(hot_sentiment, new_sentiment):

    print(f" ========= Analysis of the sentiment scores derived from VADER ========= \n\n")
    print("")
    sentiment_data = pd.concat([hot_sentiment, new_sentiment], ignore_index=True)
    sentiment_data.drop_duplicates(subset=['text'], inplace=True)

    #Plotting the distribution of the sentiment scores
    plt.figure(figsize=(10, 6))
    sns.histplot(sentiment_data['compound'], kde=True, bins=20, color='purple')
    plt.title(f'Distribution of Sentiment Scores - VADER')
    plt.xlabel('Sentiment Score')
    plt.ylabel('Frequency')
    plt.show()

model_analysis_vader(hot_vader, new_vader)

# %%
###### DOWNLOAD STOCK TICKERS #########

def download_stock_data(ticker):
    start_date = pd.to_datetime('2022-08-02')
    end_date = pd.to_datetime('2023-08-17')
    #download the stock
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

def download_ticker(ticker_list):
    for ticker in ticker_list:
        print(ticker)
        # Call the download_stock_data function to get adjusted closing prices
        stock_data = download_stock_data(ticker)
        #write the stock data to a csv file in the price_data folder
        stock_data.to_csv('price_data/top_13/'+ticker+'.csv')

#test S&P 500 since the symbol in yf is unique
sp = download_stock_data("^GSPC")
sp.head()
#write it to a csv file
sp.to_csv('price_data/top_13/S&P.csv')

#convert the tickers to a list
ticker_list = list(set(hot_ticker['ticker'].tolist()))

download_ticker(ticker_list)

# %%
####### READ THE DFs INTO A DICT #######

price_data = {}
for ticker in ticker_list:
    file_path = os.path.join('price_data','top_13' ,f'{ticker}.csv')
    if os.path.exists(file_path):
        price_data[ticker] = pd.read_csv(file_path)
    else:
        print(f"File for {ticker} does not exist!")

####### CALCULATE THE RETURNS #######

for ticker, df in price_data.items():
    close_prices = df['Close']
    for days in range(1, 6):
        df[f'Returns_{days}'] = close_prices.pct_change(periods=days)
    price_data[ticker] = df


# %%
########## MERGE THE RETURNS AND SENTIMENT SCORES ##############

def merge_dataframes(sentiment_df, price_data_dict):
    merged_rows = []
    for index, row in sentiment_df.iterrows():
        date = row['date']
        text = row['text']
        ticker = row['ticker']
        label = row['label']
        score = row['score']

        # Check if the ticker exists in price_data_ret
        if ticker in price_data_dict:
            ticker_df = price_data_dict[ticker]
        else:
            # If ticker not found, use the 'S&P' DataFrame
            ticker_df = price_data_dict['S&P']
            ticker = 'S&P'
        
        matching_date = ticker_df[ticker_df['Date'] == date]

        # If a matching date is found, merge the entire row
        if not matching_date.empty:
            op = matching_date['Open'].iloc[0]
            hi = matching_date['High'].iloc[0]
            lo = matching_date['Low'].iloc[0]
            cl = matching_date['Close'].iloc[0]
            vol = matching_date['Volume'].iloc[0]
            ret_1 = matching_date['Returns_1'].iloc[0]
            ret_2 = matching_date['Returns_2'].iloc[0]
            ret_3 = matching_date['Returns_3'].iloc[0]
            ret_4 = matching_date['Returns_4'].iloc[0]
            ret_5 = matching_date['Returns_5'].iloc[0]

            merged_rows.append([date, text, ticker,label,score, op, hi, lo, cl, vol, ret_1, ret_2, ret_3, ret_4, ret_5])

    #make new df
    merged_df = pd.DataFrame(merged_rows, columns=['date','text','ticker','label', 'score', 'open','high','low','close','vol','ret_1','ret_2','ret_3','ret_4','ret_5'])

    return merged_df

# combine the sentiment scores with the price data
hot_finbert_ret = merge_dataframes(hot_finbert,price_data)
new_finbert_ret = merge_dataframes(new_finbert,price_data)

hot_chatgpt_ret = merge_dataframes(hot_chatgpt,price_data)
new_chatgpt_ret = merge_dataframes(new_chatgpt,price_data)

hot_roberta_ret = merge_dataframes(hot_roberta,price_data)
new_roberta_ret = merge_dataframes(new_roberta,price_data)

#write the variables to save the csv
hot_finbert_ret.to_csv('sentiment_score_ret/top_13/hot_finbert_ret.csv',index=False)
new_finbert_ret.to_csv('sentiment_score_ret/top_13/new_finbert_ret.csv',index=False)
hot_chatgpt_ret.to_csv('sentiment_score_ret/top_13/hot_chatgpt_ret.csv',index=False)
new_chatgpt_ret.to_csv('sentiment_score_ret/top_13/new_chatgpt_ret.csv',index=False)
hot_roberta_ret.to_csv('sentiment_score_ret/top_13/hot_roberta_ret.csv',index=False)
new_roberta_ret .to_csv('sentiment_score_ret/top_13/new_roberta_ret.csv',index=False)

# %%
# VADER

def merge_dataframes_vader(sentiment_df, price_data_dict):
    # List to store the merged rows
    merged_rows = []

    # Iterate through the rows in sentiment_df
    for index, row in sentiment_df.iterrows():
        date = row['date']
        text = row['text']
        ticker = row['ticker']
        compound = row['compound']

        # Check if the ticker exists in price_data_ret
        if ticker in price_data_dict:
            ticker_df = price_data_dict[ticker]
        else:
            # If ticker not found, use the 'S&P' DataFrame
            ticker_df = price_data_dict['S&P']
            ticker = 'S&P'
        
        matching_date = ticker_df[ticker_df['Date'] == date]

        # If a matching date is found, merge the entire row
        if not matching_date.empty:
            op = matching_date['Open'].iloc[0]
            hi = matching_date['High'].iloc[0]
            lo = matching_date['Low'].iloc[0]
            cl = matching_date['Close'].iloc[0]
            vol = matching_date['Volume'].iloc[0]
            ret_1 = matching_date['Returns_1'].iloc[0]
            ret_2 = matching_date['Returns_2'].iloc[0]
            ret_3 = matching_date['Returns_3'].iloc[0]
            ret_4 = matching_date['Returns_4'].iloc[0]
            ret_5 = matching_date['Returns_5'].iloc[0]

            merged_rows.append([date, text, ticker,compound, op, hi, lo, cl, vol, ret_1, ret_2, ret_3, ret_4, ret_5])

    merged_df = pd.DataFrame(merged_rows, columns=['date','text','ticker','compound', 'open','high','low','close','vol','ret_1','ret_2','ret_3','ret_4','ret_5'])

    return merged_df

hot_vader_top = merge_dataframes_vader(hot_vader,price_data)
new_vader_top = merge_dataframes_vader(new_vader,price_data)

#write vader to csv file
hot_vader_top.to_csv('sentiment_score_ret/top_13/hot_vader_ret.csv',index=False)
new_vader_top.to_csv('sentiment_score_ret/top_13/new_vader_ret.csv',index=False)

# %%
############# NEURAL NETWORK MODELS ##############

import tensorflow as tf
from keras.layers import Input, Embedding, LSTM, Dense, Conv1D, MaxPooling1D, GlobalMaxPooling1D, concatenate
from keras.models import Model
from keras.utils import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from keras.preprocessing.text import Tokenizer
from tensorflow.keras import metrics
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dropout, BatchNormalization

# %%
def plot_loss_and_accuracy(history):
    fig, axs = plt.subplots(1, 2, figsize=(12,5))

    # Plot training & validation accuracy values
    axs[0].plot(history.history['accuracy'])
    axs[0].plot(history.history['val_accuracy'])
    axs[0].set_title('Model accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation loss values
    axs[1].plot(history.history['loss'])
    axs[1].plot(history.history['val_loss'])
    axs[1].set_title('Model loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].legend(['Train', 'Validation'], loc='upper left')

    plt.show()

def plot_week_eval(accuracy,precision,recall,loss):
  labels = ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5']
  fig, axes = plt.subplots(2, 2, figsize=(18, 12))
  # Accuracy
  axes[0, 0].plot(labels, accuracy, marker='o')
  axes[0, 0].set_title('Accuracy')
  axes[0, 0].grid(True)
  # Precision
  axes[0, 1].plot(labels, precision, marker='o')
  axes[0, 1].set_title('Precision')
  axes[0, 1].grid(True)
  #Recall
  axes[1, 0].plot(labels, recall, marker='o')
  axes[1, 0].set_title('Recall')
  axes[1, 0].grid(True)
  #Loss
  axes[1, 1].plot(labels, loss, marker='o')
  axes[1, 1].set_title('Loss')
  axes[1, 1].grid(True)

  plt.show()

# %%
####### LSTM #########

def LSTM_weekly(df, return_day, max_words=10000):
    # Preprocessing
    df['text'] = df['text'].astype(str)
    label_dummies = pd.get_dummies(df['label'], prefix='label')
    ticker_dummies = pd.get_dummies(df['ticker'], prefix='ticker')
    df = pd.concat([df, label_dummies, ticker_dummies], axis=1)
    df['score'] = df['score'].astype(float)

    X = df[['text'] + list(label_dummies.columns) + list(ticker_dummies.columns) + ['score', 'open', 'high', 'low', 'close', 'vol']]
    y = (df[f'ret_{return_day}'] > 0).astype(int)

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Text Tokenization and Padding
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(X_train['text'])
    train_text_sequences = tokenizer.texts_to_sequences(X_train['text'])
    test_text_sequences = tokenizer.texts_to_sequences(X_test['text'])
    max_sequence_length = max(len(s) for s in train_text_sequences + test_text_sequences)
    train_text_padded = pad_sequences(train_text_sequences, maxlen=max_sequence_length)
    test_text_padded = pad_sequences(test_text_sequences, maxlen=max_sequence_length)

    # Model Inputs
    text_input = Input(shape=(max_sequence_length,), name='text_input')
    ticker_input = Input(shape=(len(ticker_dummies.columns),), name='ticker_input')
    label_input = Input(shape=(len(label_dummies.columns),), name='label_input')
    score_input = Input(shape=(1,), name='score_input')
    market_input = Input(shape=(5, 1), name='market_input')  # Changed shape to 3D for LSTM

    # Model Layers
    text_embedding = layers.Embedding(input_dim=max_words, output_dim=128)(text_input)
    text_lstm = layers.LSTM(128)(text_embedding)
    text_dropout = Dropout(0.5)(text_lstm)

    # LSTM layer for market data
    market_lstm = layers.LSTM(32)(market_input) 
    market_dropout = Dropout(0.5)(market_lstm)

    # Concatenate all inputs
    combined_inputs = layers.concatenate([text_dropout, ticker_input, label_input, score_input, market_dropout])

    # LSTM layer
    x = layers.Reshape((combined_inputs.shape[1], 1))(combined_inputs)
    x = layers.LSTM(128)(x)
    
    # Dense layers
    x = layers.Dense(128, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = layers.Dense(68, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = Dropout(0.5)(x)
    output = layers.Dense(1, activation='sigmoid')(x)

    # Compile the model
    model = Model(inputs=[text_input, ticker_input, label_input, score_input, market_input], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', 
                  metrics=['accuracy', metrics.Precision(name='precision'), metrics.Recall(name='recall')])

    #Plot the model only one time
    if return_day == 1:
        tf.keras.utils.plot_model(model, show_shapes=True)
        model.summary()
        
    #Train the model   
    history = model.fit(
        {'text_input': train_text_padded, 'ticker_input': X_train[ticker_dummies.columns].values, 
         'label_input': X_train[label_dummies.columns].values, 'score_input': X_train['score'].values,
         'market_input': X_train[['open', 'high', 'low', 'close', 'vol']].values.reshape(-1, 5, 1)},  # Reshaped for LSTM
        y_train,
        epochs=20,
        batch_size=32,
        validation_split=0.2
    )

    plot_loss_and_accuracy(history)

    # Evaluate the model
    evaluation = model.evaluate(
        {'text_input': test_text_padded, 'ticker_input': X_test[ticker_dummies.columns].values, 
         'label_input': X_test[label_dummies.columns].values, 'score_input': X_test['score'].values,
         'market_input': X_test[['open', 'high', 'low', 'close', 'vol']].values.reshape(-1, 5, 1)},  # Reshaped for LSTM
        y_test
    )
    print("Test Accuracy for Returns from day ", return_day, " : {:.2f}%".format(evaluation[1] * 100))
    print("Test Precision for Returns from day ", return_day, " : {:.2f}%".format(evaluation[2] * 100))
    print("Test Recall for Returns from day ", return_day, " : {:.2f}%".format(evaluation[3] * 100))
    print("Test Loss for Returns from day ", return_day, " : {:.4f}".format(evaluation[0]))

    return evaluation


# %%
print("=============  HOT FINBERT LSTM  =============== \n")
accuracy =[]
precision=[]
recall =[]
loss=[]
for i in range(1,6):
  evaluation = LSTM_weekly(hot_finbert,i)
  loss.append(evaluation[0])
  accuracy.append(evaluation[1])
  precision.append(evaluation[2]) 
  recall.append(evaluation[3])

plot_week_eval(accuracy,precision,recall,loss)

print("=============  NEW FINBERT LSTM  =============== \n")
accuracy =[]
precision=[]
recall =[]
loss=[]
for i in range(1,6):
  evaluation = LSTM_weekly(new_finbert,i)
  loss.append(evaluation[0])
  accuracy.append(evaluation[1])
  precision.append(evaluation[2]) 
  recall.append(evaluation[3])

plot_week_eval(accuracy,precision,recall,loss)

# %%
print("=============  HOT ROBERTA LSTM  =============== \n")
accuracy =[]
precision=[]
recall =[]
loss=[]
for i in range(1,6):
  evaluation = LSTM_weekly(hot_roberta,i)
  loss.append(evaluation[0])
  accuracy.append(evaluation[1])
  precision.append(evaluation[2]) 
  recall.append(evaluation[3])

plot_week_eval(accuracy,precision,recall,loss)

print("=============  NEW ROBERTA LSTM  =============== \n")
accuracy =[]
precision=[]
recall =[]
loss=[]
for i in range(1,6):
  evaluation = LSTM_weekly(new_roberta,i)
  loss.append(evaluation[0])
  accuracy.append(evaluation[1])
  precision.append(evaluation[2]) 
  recall.append(evaluation[3])

plot_week_eval(accuracy,precision,recall,loss)

# %%
print("=============  HOT GPT-3 LSTM  =============== \n")
accuracy =[]
precision=[]
recall =[]
loss=[]
for i in range(1,6):
  evaluation = LSTM_weekly(hot_chatgpt,i)
  loss.append(evaluation[0])
  accuracy.append(evaluation[1])
  precision.append(evaluation[2]) 
  recall.append(evaluation[3])

plot_week_eval(accuracy,precision,recall,loss)

print("=============  NEW GPT-3 LSTM  =============== \n")
accuracy =[]
precision=[]
recall =[]
loss=[]
for i in range(1,6):
  evaluation = LSTM_weekly(new_chatgpt,i)
  loss.append(evaluation[0])
  accuracy.append(evaluation[1])
  precision.append(evaluation[2]) 
  recall.append(evaluation[3])

plot_week_eval(accuracy,precision,recall,loss)

# %%
# VADER

def LSTM_weekly_vader(df, return_day, max_words=10000):
    # Preprocessing
    df['text'] = df['text'].astype(str)
    ticker_dummies = pd.get_dummies(df['ticker'], prefix='ticker')
    df = pd.concat([df, ticker_dummies], axis=1)
    
    # Feature and Target Variables
    X = df[['text'] + list(ticker_dummies.columns) + ['compound', 'open', 'high', 'low', 'close', 'vol']]
    y = (df[f'ret_{return_day}'] > 0).astype(int)

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Text Tokenization and Padding
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(X_train['text'])
    train_text_sequences = tokenizer.texts_to_sequences(X_train['text'])
    test_text_sequences = tokenizer.texts_to_sequences(X_test['text'])
    max_sequence_length = max(len(s) for s in train_text_sequences + test_text_sequences)
    train_text_padded = pad_sequences(train_text_sequences, maxlen=max_sequence_length)
    test_text_padded = pad_sequences(test_text_sequences, maxlen=max_sequence_length)

    # Model Inputs
    text_input = Input(shape=(max_sequence_length,), name='text_input')
    ticker_input = Input(shape=(len(ticker_dummies.columns),), name='ticker_input')
    compound_input = Input(shape=(1,), name='compound_input')
    market_input = Input(shape=(5,), name='market_input')

    # Model Layers
    text_embedding = layers.Embedding(input_dim=max_words, output_dim=128)(text_input)
    text_lstm = layers.LSTM(128)(text_embedding)
    text_dropout = Dropout(0.5)(text_lstm)
    
    market_dense = layers.Dense(32, activation='relu', kernel_regularizer=l2(0.001))(market_input)
    market_dropout = Dropout(0.5)(market_dense)

    combined_inputs = layers.concatenate([text_dropout, ticker_input, compound_input, market_dropout])
    
    x = layers.Dense(128, activation='relu', kernel_regularizer=l2(0.001))(combined_inputs)
    x = BatchNormalization()(x)
    x = layers.Dense(68, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = Dropout(0.5)(x)
    output = layers.Dense(1, activation='sigmoid')(x)

    # Compile and Train Model
    model = Model(inputs=[text_input, ticker_input, compound_input, market_input], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', 
                  metrics=['accuracy', metrics.Precision(name='precision'), metrics.Recall(name='recall')])
    
    if return_day == 1:
        tf.keras.utils.plot_model(model, show_shapes=True)
        model.summary()
        
    history = model.fit(
        {'text_input': train_text_padded, 'ticker_input': X_train[ticker_dummies.columns].values, 
         'compound_input': X_train['compound'].values,
         'market_input': X_train[['open', 'high', 'low', 'close', 'vol']].values},
        y_train,
        epochs=20,
        batch_size=32,
        validation_split=0.2
    )
    plot_loss_and_accuracy(history)
    # Evaluate the model
    evaluation = model.evaluate(
        {'text_input': test_text_padded, 'ticker_input': X_test[ticker_dummies.columns].values, 
         'compound_input': X_test['compound'].values,
         'market_input': X_test[['open', 'high', 'low', 'close', 'vol']].values},
        y_test
    )

    print("Test Accuracy for Returns from day ", return_day, " : {:.2f}%".format(evaluation[1] * 100))
    print("Test Precision for Returns from day ", return_day, " : {:.2f}%".format(evaluation[2] * 100))
    print("Test Recall for Returns from day ", return_day, " : {:.2f}%".format(evaluation[3] * 100))
    print("Test Loss for Returns from day ", return_day, " : {:.4f}".format(evaluation[0]))

    return evaluation


# %%
print("=============  HOT VADER LSTM  =============== \n")
accuracy =[]
precision=[]
recall =[]
loss=[]
for i in range(1,6):
  evaluation = LSTM_weekly_vader(hot_vader,i)
  loss.append(evaluation[0])
  accuracy.append(evaluation[1])
  precision.append(evaluation[2]) 
  recall.append(evaluation[3])

plot_week_eval(accuracy,precision,recall,loss)

print("=============  NEW VADER LSTM  =============== \n")
accuracy =[]
precision=[]
recall =[]
loss=[]
for i in range(1,6):
  evaluation = LSTM_weekly_vader(new_vader,i)
  loss.append(evaluation[0])
  accuracy.append(evaluation[1])
  precision.append(evaluation[2]) 
  recall.append(evaluation[3])

plot_week_eval(accuracy,precision,recall,loss)

# %%
############ CNN #############

def CNN_weekly(df, return_day, max_words=10000):

    #Preprocessing
    df['text'] = df['text'].astype(str)
    df['score'] = df['score'].astype(float)
    label_dummies = pd.get_dummies(df['label'], prefix='label')
    ticker_dummies = pd.get_dummies(df['ticker'], prefix='ticker')
    df = pd.concat([df, label_dummies, ticker_dummies], axis=1)
    
    X = df[['text'] + list(label_dummies.columns) + list(ticker_dummies.columns) + ['score', 'open', 'high', 'low', 'close', 'vol']]
    y = (df[f'ret_{return_day}'] > 0).astype(int)

    # Split the data into a training set and a testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #Text Tokenization and Padding
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(X_train['text'])
    train_text_sequences = tokenizer.texts_to_sequences(X_train['text'])
    test_text_sequences = tokenizer.texts_to_sequences(X_test['text'])
    max_sequence_length = max(len(s) for s in train_text_sequences + test_text_sequences)
    train_text_padded = pad_sequences(train_text_sequences, maxlen=max_sequence_length)
    test_text_padded = pad_sequences(test_text_sequences, maxlen=max_sequence_length)

    # Model Inputs
    text_input = Input(shape=(max_sequence_length,), name='text_input')
    ticker_input = Input(shape=(len(ticker_dummies.columns),), name='ticker_input')
    label_input = Input(shape=(len(label_dummies.columns),), name='label_input')
    score_input = Input(shape=(1,), name='score_input')
    market_input = Input(shape=(5, 1), name='market_input')

    #Convert to Embedding and define the text layers
    text_embedding = layers.Embedding(input_dim=max_words, output_dim=128)(text_input)
    text_conv = Conv1D(128, 3, activation='relu')(text_embedding)
    text_pool = MaxPooling1D(3)(text_conv)
    text_flat = GlobalMaxPooling1D()(text_pool)
    text_dropout = Dropout(0.5)(text_flat)
    # Market layers
    market_conv1 = Conv1D(32, 2, activation='relu')(market_input)
    market_pool1 = MaxPooling1D(2)(market_conv1)
    market_flat = GlobalMaxPooling1D()(market_pool1)
    market_dropout = Dropout(0.5)(market_flat)

    #Concatenate the Inputs
    combined_inputs = layers.concatenate([text_dropout, ticker_input, label_input, score_input, market_dropout])

    #Dense layers
    x = layers.Dense(128, activation='relu', kernel_regularizer=l2(0.001))(combined_inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(68, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = Dropout(0.5)(x)
    output = layers.Dense(1, activation='sigmoid')(x)

    #Compile the model
    model = Model(inputs=[text_input, ticker_input, label_input, score_input, market_input], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy', metrics.Precision(name='precision'), metrics.Recall(name='recall')])

    #Plot the model only one time
    if return_day == 1:
        tf.keras.utils.plot_model(model, show_shapes=True)
        model.summary()

    #Train the model
    history = model.fit(
        {'text_input': train_text_padded, 'ticker_input': X_train[ticker_dummies.columns].values,
         'label_input': X_train[label_dummies.columns].values, 'score_input': X_train['score'].values,
         'market_input': X_train[['open', 'high', 'low', 'close', 'vol']].values.reshape(-1, 5, 1)},
        y_train,
        epochs=20,
        batch_size=32,
        validation_split=0.2
    )

    plot_loss_and_accuracy(history)

    # Evaluate the model
    evaluation = model.evaluate(
        {'text_input': test_text_padded, 'ticker_input': X_test[ticker_dummies.columns].values,
         'label_input': X_test[label_dummies.columns].values, 'score_input': X_test['score'].values,
         'market_input': X_test[['open', 'high', 'low', 'close', 'vol']].values.reshape(-1, 5, 1)},
        y_test
    )
    
    print("Test Accuracy for Returns from day ", return_day, " : {:.2f}%".format(evaluation[1] * 100))
    print("Test Precision for Returns from day ", return_day, " : {:.2f}%".format(evaluation[2] * 100))
    print("Test Recall for Returns from day ", return_day, " : {:.2f}%".format(evaluation[3] * 100))
    print("Test Loss for Returns from day ", return_day, " : {:.4f}".format(evaluation[0]))

    return evaluation


# %%
print("=============  HOT FINBERT CNN  =============== \n")
accuracy =[]
precision=[]
recall =[]
loss=[]
for i in range(1,6):
  evaluation = CNN_weekly(hot_finbert,i)
  loss.append(evaluation[0])
  accuracy.append(evaluation[1])
  precision.append(evaluation[2])
  recall.append(evaluation[3])

plot_week_eval(accuracy,precision,recall,loss)

print("=============  NEW FINBERT CNN  =============== \n")
accuracy =[]
precision=[]
recall =[]
loss=[]
for i in range(1,6):
  evaluation = CNN_weekly(new_finbert,i)
  loss.append(evaluation[0])
  accuracy.append(evaluation[1])
  precision.append(evaluation[2])
  recall.append(evaluation[3])

plot_week_eval(accuracy,precision,recall,loss)

# %%
print("=============  HOT ROBERTA CNN  =============== \n")
accuracy =[]
precision=[]
recall =[]
loss=[]
for i in range(1,6):
  evaluation = CNN_weekly(hot_roberta,i)
  loss.append(evaluation[0])
  accuracy.append(evaluation[1])
  precision.append(evaluation[2])
  recall.append(evaluation[3])

plot_week_eval(accuracy,precision,recall,loss)

print("=============  NEW ROBERTA CNN  =============== \n")
accuracy =[]
precision=[]
recall =[]
loss=[]
for i in range(1,6):
  evaluation = CNN_weekly(new_roberta,i)
  loss.append(evaluation[0])
  accuracy.append(evaluation[1])
  precision.append(evaluation[2])
  recall.append(evaluation[3])

plot_week_eval(accuracy,precision,recall,loss)

# %%
print("=============  HOT GPT-3 CNN  =============== \n")
accuracy =[]
precision=[]
recall =[]
loss=[]
for i in range(1,6):
  evaluation = CNN_weekly(hot_chatgpt,i)
  loss.append(evaluation[0])
  accuracy.append(evaluation[1])
  precision.append(evaluation[2])
  recall.append(evaluation[3])

plot_week_eval(accuracy,precision,recall,loss)

print("=============  NEW GPT-3 CNN  =============== \n")
accuracy =[]
precision=[]
recall =[]
loss=[]
for i in range(1,6):
  evaluation = CNN_weekly(new_chatgpt,i)
  loss.append(evaluation[0])
  accuracy.append(evaluation[1])
  precision.append(evaluation[2])
  recall.append(evaluation[3])

plot_week_eval(accuracy,precision,recall,loss)

# %%
def CNN_weekly_vader(df, return_day, max_words=10000):
    # Preprocessing
    df['text'] = df['text'].astype(str)
    ticker_dummies = pd.get_dummies(df['ticker'], prefix='ticker')
    df = pd.concat([df, ticker_dummies], axis=1)
    df['compound'] = df['compound'].astype(float)

    # Feature and Target Variables
    X = df[['text'] + list(ticker_dummies.columns) + ['compound', 'open', 'high', 'low', 'close', 'vol']]
    y = (df[f'ret_{return_day}'] > 0).astype(int)

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Text Tokenization and Padding
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(X_train['text'])
    train_text_sequences = tokenizer.texts_to_sequences(X_train['text'])
    test_text_sequences = tokenizer.texts_to_sequences(X_test['text'])
    max_sequence_length = max(len(s) for s in train_text_sequences + test_text_sequences)
    train_text_padded = pad_sequences(train_text_sequences, maxlen=max_sequence_length)
    test_text_padded = pad_sequences(test_text_sequences, maxlen=max_sequence_length)

    # Model Inputs
    text_input = Input(shape=(max_sequence_length,), name='text_input')
    ticker_input = Input(shape=(len(ticker_dummies.columns),), name='ticker_input')
    compound_input = Input(shape=(1,), name='compound_input')
    market_input = Input(shape=(5,), name='market_input')

    # Model Layers
    text_embedding = layers.Embedding(input_dim=max_words, output_dim=128)(text_input)
    text_conv = Conv1D(128, 3, activation='relu')(text_embedding)
    text_pool = MaxPooling1D(3)(text_conv)
    text_flat = GlobalMaxPooling1D()(text_pool)
    text_dropout = Dropout(0.5)(text_flat)

    market_dense = layers.Dense(32, activation='relu', kernel_regularizer=l2(0.001))(market_input)
    market_dropout = Dropout(0.5)(market_dense)

    combined_inputs = layers.concatenate([text_dropout, ticker_input, compound_input, market_dropout])

    x = layers.Dense(128, activation='relu', kernel_regularizer=l2(0.001))(combined_inputs)
    x = BatchNormalization()(x)
    x = layers.Dense(68, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = Dropout(0.5)(x)
    output = layers.Dense(1, activation='sigmoid')(x)

    # Compile and Train Model
    model = Model(inputs=[text_input, ticker_input, compound_input, market_input], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy', metrics.Precision(name='precision'), metrics.Recall(name='recall')])

    if return_day == 1:
        tf.keras.utils.plot_model(model, show_shapes=True)
        model.summary()

    #Train the model
    history = model.fit(
        {'text_input': train_text_padded, 'ticker_input': X_train[ticker_dummies.columns].values,
         'compound_input': X_train['compound'].values,
         'market_input': X_train[['open', 'high', 'low', 'close', 'vol']].values},
        y_train,
        epochs=10,
        batch_size=32,
        validation_split=0.2
    )

    plot_loss_and_accuracy(history)

    # Evaluate the model
    evaluation = model.evaluate(
        {'text_input': test_text_padded, 'ticker_input': X_test[ticker_dummies.columns].values,
         'compound_input': X_test['compound'].values,
         'market_input': X_test[['open', 'high', 'low', 'close', 'vol']].values},
        y_test
    )
    print("Test Accuracy for Returns from day ", return_day, " : {:.2f}%".format(evaluation[1] * 100))
    print("Test Precision for Returns from day ", return_day, " : {:.2f}%".format(evaluation[2] * 100))
    print("Test Recall for Returns from day ", return_day, " : {:.2f}%".format(evaluation[3] * 100))
    print("Test Loss for Returns from day ", return_day, " : {:.4f}".format(evaluation[0]))

    return evaluation

# %%
print("=============  HOT VADER CNN  =============== \n")
accuracy =[]
precision=[]
recall =[]
loss=[]
for i in range(1,6):
  evaluation = CNN_weekly_vader(hot_vader,i)
  loss.append(evaluation[0])
  accuracy.append(evaluation[1])
  precision.append(evaluation[2])
  recall.append(evaluation[3])

plot_week_eval(accuracy,precision,recall,loss)

print("=============  NEW VADER CNN  =============== \n")
accuracy =[]
precision=[]
recall =[]
loss=[]
for i in range(1,6):
  evaluation = CNN_weekly_vader(new_vader,i)
  loss.append(evaluation[0])
  accuracy.append(evaluation[1])
  precision.append(evaluation[2])
  recall.append(evaluation[3])

plot_week_eval(accuracy,precision,recall,loss)


