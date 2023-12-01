# %% [markdown]
# # Draft 2 - Dissertation
# 
# Sentiment analysis to understand the effect of meme community on the stock market. This paper studies how r/wallstreetbets have an impact on the stock market and how some companies are discussed more due to the relevance/popularity of the company in the general space.

# %% [markdown]
# # Importing Libraries
# 
# TODO: ML libraries to be added

# %%
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
import time
import seaborn as sns
from scipy.stats import norm


# %% [markdown]
# # Read the symbols into a variable 
# 
# and search any specific symbol, and/or add "$" to a paticular symbol

# %%
symbols = pd.read_csv('symbols/symbols.csv')
symbols.head(3)

# %%
#search for a specific symbol
sym = '$NOW'
print(symbols.loc[symbols['Symbol'] == sym])

# %%
#add a "$" to a specific Symbol column in the symbols df from the change variable matching the Symbols column
change = 'NEXT'
symbols.loc[symbols['Symbol'] == change, 'Symbol'] = '$' + change
print(symbols.loc[symbols['Symbol'] == '$'+change])
#save the symbols df in the csv file
symbols.to_csv('symbols/symbols.csv', index=False)

# %% [markdown]
# # Cleaning the data - Remove duplicates
# 

# %%
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
# File inputs
input_file = 'raw_txt/hot.txt'
output_file = 'clean_txt/clean_hot.txt'

remove_duplicates(input_file, output_file)

# %%
# File inputs
input_file = 'raw_txt/new.txt'
output_file = 'clean_txt/clean_new.txt'

remove_duplicates(input_file, output_file)

# %% [markdown]
# # Read the hot file into a variable

# %%
#import the hot text file to a list
hot = open('clean_txt/clean_hot.txt').read().split('\n')
#count the number of items in the list
len(hot)

# %%
new = open('clean_txt/clean_new.txt').read().split('\n')
len(new)

# %% [markdown]
# # Make the lines into a dataframe with dates and text
# 
# this is done so it is easy to get the dates with the sentences

# %%
def make_dataframe(file_name):
    # Initialize a list to store the DataFrame rows
    rows = []

    # Open the file for reading
    with open(file_name, 'r') as file:
        current_date = None

        # Read the file line by line
        for line in file:
            # Check if the line matches the date format
            match = re.search(r'[-]+(\d{4}-\d{2}-\d{2})', line)
            if match:
                # Extract the date in the format "YYYY-MM-DD"
                current_date = match.group(1) # Only extract the date part without dashes

            else:
                # If the line is not a date, add it as a row in the DataFrame
                if current_date is not None: # Avoid adding rows without an associated date
                    rows.append({'date': current_date, 'text': line.strip()})

    # Create a DataFrame from the list of rows
    df = pd.DataFrame(rows, columns=['date', 'text'])

    # Print the DataFrame to see the results
    print(df)

    # Return the DataFrame
    return df

# %%
hot_df = make_dataframe('clean_txt/clean_hot.txt')

# %%
new_df[new_df['text'] == ""]

# %%
new_df = make_dataframe('clean_txt/clean_new.txt')

# %%
#write hot_df into a csv file named hot_df with no index
hot_df.to_csv('text_df/hot_df.csv', index=False)
new_df.to_csv('text_df/new_df.csv', index=False)

# %%
hot_df = pd.read_csv('text_df/hot_df.csv')
print(f"Hot length: {len(hot_ticker)}")
new_df = pd.read_csv('text_df/new_df.csv')
print(f"New length: {len(new_ticker)}")

# %% [markdown]
# # Find the tickers in each sentence

# %%
def symbol_matches(sentence, symbol_list):
    matches = []
    for symbol in symbol_list:
        if pd.notna(symbol):  # Skip NaN values in the symbol_list
            pattern = r"\b" + re.escape(str(symbol)) + r"\b"
            if re.search(pattern, sentence):
                matches.append(symbol)
    return matches

def find_matches(hot_df, symbols):
    # List to store the results
    results = []

    # Iterate through each row in the DataFrame
    for index, row in hot_df.iterrows():
        date = row['date']
        sentence = row['text']

        # Find matches for each sentence with "Symbol" and "Dollar_Symbol" columns
        matches_symbols = symbol_matches(sentence, symbols['Symbol'].tolist())
        matches_dollar_symbols = symbol_matches(sentence, symbols['Dollar_Symbol'].tolist())
        all_matches = matches_symbols + matches_dollar_symbols

        # If all_matches is empty then input "S&P"
        if not all_matches:
            all_matches = ['S&P']

        # Concatenate the matches into a single string
        ticker_string = ', '.join(all_matches)

        # Append the results to the list
        results.append((date, sentence, ticker_string))

    # Create a new DataFrame to store the results
    tickers = pd.DataFrame(results, columns=["date", "text", "ticker"])

    # Print the result DataFrame
    print(tickers)

    # Return the result DataFrame  
    return tickers


# %%
hot_ticker = find_matches(hot_df,symbols)

# %%
new_ticker = find_matches(new_df,symbols)

# %%
# Filter the DataFrame to exclude rows with 'S&P' in the 'Ticker' column
non_sp_sentences = hot_ticker[hot_ticker['ticker'] != 'S&P']
len(non_sp_sentences)


# %%
sp_sentences = hot_ticker[hot_ticker['ticker'] == 'S&P']
len(sp_sentences)

# %%
#write hot_ticker to a csv with no index
hot_ticker.to_csv('text_df/hot_ticker.csv', index=False)
new_ticker.to_csv('text_df/new_ticker.csv', index=False)

# %%
#read hot_ticker csv to a variable name hot_ticker
hot_ticker = pd.read_csv('text_df/hot_ticker.csv')
print(f"Hot length: {len(hot_ticker)}")
new_ticker = pd.read_csv('text_df/new_ticker.csv')
print(f"New length: {len(new_ticker)}")

# %% [markdown]
# # FINBERT

# %%
# Use a pipeline as a high-level helper
from transformers import pipeline

pipe_finbert = pipeline("text-classification", model="ProsusAI/finbert")

# %%
def finbert_sentiments(df,pipe=pipe_finbert):
    # Create an empty list to store the rows
    rows = []
    
    # Iterate through the DataFrame rows
    for index, row in df.iterrows():
        date = row['date']
        text = row['text']
        ticker = row['ticker']

        # If text is a string, convert it to a list with one element
        if isinstance(text, str):
            sentences = [text]
        
        # Process the sentences through the pipeline
        results = pipe(sentences)
        
        # Iterate through the sentences and results to create rows
        for sentence, result in zip(sentences, results):
            rows.append((date, sentence, ticker, result['label'], result['score']))
            
    # Create a DataFrame from the rows
    finbert_sentiment = pd.DataFrame(rows, columns=['date', 'text', 'ticker', 'label', 'score'])
    
    return finbert_sentiment


# %%
hot_finbert = finbert_sentiments(hot_ticker)
print(hot_finbert.head(5))
print(f"Hot Finbert length: {len(hot_ticker)}")

# %%
new_finbert = finbert_sentiments(new_ticker)
print(new_finbert.head(5))
print(f"New Finbert length: {len(new_ticker)}")

# %%
#write hot_sentiments into a csv with no index
hot_finbert.to_csv('sentiment_scores/hot_finbert.csv', index=False)
new_finbert.to_csv('sentiment_scores/new_finbert.csv', index=False)

# %%
hot_finbert[hot_finbert['ticker'].str.contains(',')]

# %%
new_finbert[new_finbert['ticker'].str.contains(',')]

# %% [markdown]
# # VADER

# %%
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer_vader = SentimentIntensityAnalyzer()

# %%
def vader_sentiments(df,analyzer=analyzer_vader):
    # Create an empty list to store the rows
    rows = []
    
    # Iterate through the keys and values in hot_dict
    for index, row in df.iterrows():
        date = row['date']
        text = row['text']
        ticker = row['ticker']
        # If sentences is a string, convert it to a list with one element
        if isinstance(text, str):
            sentences = [text]
        
        # Process the sentences through the pipeline
        for sentence in sentences:
            result = analyzer.polarity_scores(sentence)
            rows.append((date, sentence, ticker, result['neg'], result['neu'], result['pos'], result['compound']))
            
    # Create a DataFrame from the rows
    vader_sentiment = pd.DataFrame(rows, columns=['date', 'text', 'ticker', 'neg', 'neu', 'pos', 'compound'])
    
    return vader_sentiment

# %%
hot_vader = vader_sentiments(hot_ticker)
print(hot_vader.head(10))
print(f"Lenght of Hot Vader: {len(hot_vader)}")

# %%
new_vader = vader_sentiments(new_ticker)
print(new_vader.head(10))
print(f"Lenght of Hot Vader: {len(new_vader)}")

# %%
#write hot_vader to csv file with no index
hot_vader.to_csv('sentiment_scores/hot_vader.csv', index=False)
new_vader.to_csv('sentiment_scores/new_vader.csv', index=False)

# %% [markdown]
# # ChatGPT

# %%
load_dotenv()
openai.api_key = getenv('OPENAI_API_KEY')


# %%
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

# %%
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
# print all rows after row 697 in hot_ticker
hot_chatgpt = chatgpt_sentiments(hot_ticker)
len(hot_chatgpt)

# %%
hot_chatgpt[hot_chatgpt['label']=='error']

# %%
tmp = chatgpt_sentiments(hot_chatgpt[hot_chatgpt['label']=='error'])

# %%
merged_df = pd.merge(hot_chatgpt, tmp, on='text', suffixes=('_hot', '_tmp'))

# Iterate through the merged DataFrame and update the label and score in hot_chatgpt
for index, row in merged_df.iterrows():
    hot_index = hot_chatgpt[hot_chatgpt['text'] == row['text']].index.item()
    hot_chatgpt.at[hot_index, 'label'] = row['label_tmp']
    hot_chatgpt.at[hot_index, 'score'] = row['score_tmp']


# %%
hot_chatgpt[hot_chatgpt['label'] == 'error']

# %% [markdown]
# remove errors from Hot_chatgpt

# %%
#remove the rows from hot_chatgpt where the label is 'error'
hot_chatgpt = hot_chatgpt[hot_chatgpt['label'] != 'error']

# %%
# write hot_chatgpt to hot_chatgpt.csv with no index using pandas
hot_chatgpt.to_csv('sentiment_scores/hot_chatgpt.csv', index=False)

# %%
#read the hot_chatgpt variable
hot_chatgpt = pd.read_csv('sentiment_scores/hot_chatgpt.csv')
hot_chatgpt.head(10)


# %%
new_chatgpt = chatgpt_sentiments(new_ticker)

# %%
new_tmp = chatgpt_sentiments(new_chatgpt[new_chatgpt['score']==2])

# %%
new_merged_df = pd.merge(new_chatgpt, new_tmp, on='text', suffixes=('_new', '_tmp'))

# Iterate through the merged DataFrame and update the label and score in hot_chatgpt
for index, row in new_merged_df.iterrows():
    new_index = new_chatgpt[new_chatgpt['text'] == row['text']].index.item()
    new_chatgpt.at[new_index, 'label'] = row['label_tmp']
    new_chatgpt.at[new_index, 'score'] = row['score_tmp']

# %%
new_chatgpt[new_chatgpt['score']==2]

# %%
new_chatgpt = new_chatgpt[new_chatgpt['label']!='error']

# %%
new_chatgpt.tail(4)

# %%
new_chatgpt.to_csv('sentiment_scores/new_chatgpt.csv', index=False)

# %%
new_chatgpt = pd.read_csv('sentiment_scores/new_chatgpt.csv')
new_chatgpt.head(10)

# %%
new_chatgpt['label'] = new_chatgpt['label'].replace({
    'Positive': 'positive',
    'Neutral': 'neutral',
    'Negative': 'negative',
    'No sentiment': 'error',
    'neither': 'error'
})

# %%
hot_chatgpt['label'] = hot_chatgpt['label'].replace({
    'Positive': 'positive',
    'Neutral': 'neutral',
    'Negative': 'negative',
    'No sentiment': 'error',
    'neither': 'error'
})

# %%
print(hot_chatgpt[hot_chatgpt['label']=='error'])
print(new_chatgpt[new_chatgpt['label']=='error'])

# %% [markdown]
# # ChatGPT calculations

# %%
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize

# Make sure to download the NLTK tokenizer models
nltk.download('punkt')


# %%
def count_tokens(df):
    total = 0
    # Iterate through the DataFrame rows
    for index, row in df.iterrows():
        text = row['text']
        
        # Tokenize the text using NLTK
        tokens = word_tokenize(text)
        
        # Count the number of tokens
        num_tokens = len(tokens)
        
        total += num_tokens
        # Print the number of tokens
        print(f"Number of tokens in row {index}: {num_tokens}")
    print(f" Total number of tokens {total}")
    return total

# %%
tokens = count_tokens(hot_df)

# %%
tokensgpt = tokens /1000
print(f" Charged tokens {tokensgpt}")
price_gpt3 = 0.0015
print(f" Price per token for GPT_3 4K {price_gpt3}")
print(f" Total cost ${tokensgpt * price_gpt3}")
iterations = 50
print(f" Iterations cost ${(tokensgpt * price_gpt3)*iterations}")

# %% [markdown]
# # DISTIL ROBERTA

# %%
# Use a pipeline as a high-level helper
from transformers import pipeline

pipe_roberta = pipeline("text-classification", model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")

# %%
def roberta_sentiments(df,pipe=pipe_roberta):
    # Create an empty list to store the rows
    rows = []
    
    # Iterate through the DataFrame rows
    for index, row in df.iterrows():
        date = row['date']
        text = row['text']
        ticker = row['ticker']

        # If text is a string, convert it to a list with one element
        if isinstance(text, str):
            sentences = [text]
        
        # Process the sentences through the pipeline
        results = pipe(sentences)
        
        # Iterate through the sentences and results to create rows
        for sentence, result in zip(sentences, results):
            rows.append((date, sentence, ticker, result['label'], result['score']))
            
    # Create a DataFrame from the rows
    roberta_sentiment = pd.DataFrame(rows, columns=['date', 'text', 'ticker', 'label', 'score'])
    
    return roberta_sentiment


# %%
hot_roberta = roberta_sentiments(hot_ticker)
print(hot_roberta.head())
new_roberta = roberta_sentiments(new_ticker)
print(new_roberta.head())

# %%
#write sentiments into a csv with no index
hot_roberta.to_csv('sentiment_scores/hot_roberta.csv', index=False)
new_roberta.to_csv('sentiment_scores/new_roberta.csv', index=False)

# %% [markdown]
# # BASIC text analysis

# %%
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

def model_analysis(hot_sentiment, new_sentiment, model):

    print(f" ========= Analysis of the sentiment scores derived from {model} ========= \n\n")
    sentiment_data = pd.concat([hot_sentiment, new_sentiment], ignore_index=True)
    sentiment_data.drop_duplicates(subset=['text'], inplace=True)

    sentiment_counts = sentiment_data['label'].value_counts(normalize=True) * 100

    # Plotting the overall sentiment distribution
    plt.figure(figsize=(5, 5))
    sentiment_counts.plot.pie(autopct='%1.1f%%',colors=['lightblue', 'red', 'green'])
    plt.title('Overall Sentiment Distribution')
    plt.show()

    top_6 = sentiment_data['ticker'].value_counts().head(6)
    top_5_no_sp = top_6[1:6]

    top_5_ticker_analysis = {}
    for ticker in top_5_no_sp.index:
        top_5_ticker_analysis[ticker] = (sentiment_data[sentiment_data['ticker'] == ticker]['label'].value_counts(normalize=True) * 100).to_dict()

    top_5_ticker_df = pd.DataFrame.from_dict(top_5_ticker_analysis, orient='index').fillna(0)

    # Plotting the sentiment distribution for the new top 5 tickers (excluding "S&P")
    top_5_ticker_df.plot(kind='bar', stacked=True, figsize=(10, 6), color=['lightblue', 'red', 'green'])
    plt.title('Sentiment Distribution for Top 5 Tickers (Excluding S&P)')
    plt.xlabel('Ticker')
    plt.ylabel('Percentage')
    plt.xticks(rotation=45)
    plt.show()

    #Plotting the distribution of the sentiment scores
    plt.figure(figsize=(10, 6))
    sns.histplot(sentiment_data['score'], kde=True, bins=20, color='purple')
    plt.title('Distribution of Sentiment Scores')
    plt.xlabel('Sentiment Score')
    plt.ylabel('Frequency')
    plt.show()

    # Using CountVectorizer to create a word matrix and removing stop words
    vectorizer = CountVectorizer(stop_words='english', max_features=10)
    word_matrix_vectorized = vectorizer.fit_transform(sentiment_data['text'])

    # Calculating the correlation matrix using CountVectorizer
    correlation_matrix_vectorized = np.corrcoef(word_matrix_vectorized.toarray(), rowvar=False)

    # Plotting the correlation matrix using CountVectorizer
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix_vectorized, annot=True, cmap='coolwarm', xticklabels=vectorizer.get_feature_names_out(), yticklabels=vectorizer.get_feature_names_out())
    plt.title('Correlation Matrix of Top 10 Words (Using CountVectorizer)')
    plt.show()




# %%
from sklearn.feature_extraction.text import CountVectorizer

def model_analysis_vader(hot_sentiment, new_sentiment, model):

    print(f" ========= Analysis of the sentiment scores derived from {model} ========= \n\n")
    sentiment_data = pd.concat([hot_sentiment, new_sentiment], ignore_index=True)
    sentiment_data.drop_duplicates(subset=['text'], inplace=True)

    #Plotting the distribution of the sentiment scores
    plt.figure(figsize=(10, 6))
    sns.histplot(sentiment_data['compound'], kde=True, bins=20, color='purple')
    plt.title('Distribution of Sentiment Scores')
    plt.xlabel('Sentiment Score')
    plt.ylabel('Frequency')
    plt.show()

    # Using CountVectorizer to create a word matrix and removing stop words
    vectorizer = CountVectorizer(stop_words='english', max_features=10)
    word_matrix_vectorized = vectorizer.fit_transform(sentiment_data['text'])

    # Calculating the correlation matrix using CountVectorizer
    correlation_matrix_vectorized = np.corrcoef(word_matrix_vectorized.toarray(), rowvar=False)

    # Plotting the correlation matrix using CountVectorizer
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix_vectorized, annot=True, xticklabels=vectorizer.get_feature_names_out(), yticklabels=vectorizer.get_feature_names_out(),cmap='coolwarm')
    plt.title('Correlation Matrix of Top 10 Words (Using CountVectorizer)')
    plt.show()




# %%
model_analysis(hot_finbert, new_finbert, 'Finbert')

# %%
model_analysis(hot_chatgpt, new_chatgpt, 'ChatGPT')

# %%
model_analysis(hot_roberta, new_roberta, 'Roberta')

# %%
model_analysis_vader(hot_vader, new_vader, 'Vader')

# %% [markdown]
# # Price download and calculations
# 

# %%
def price_tickers(ticker_df):
    # Count the occurrences of each ticker and sort them in descending order
    ticker_counts_df = ticker_df['ticker'].value_counts().sort_values(ascending=False).reset_index()

    # Reset the index and rename the columns for clarity
    ticker_counts_df.columns = ['Ticker', 'Count']
    # Filter the tickers that occur more than three times
    frequent_tickers_df = ticker_counts_df[ticker_counts_df['Count'] > 3]
    print(frequent_tickers_df)
    # Convert the filtered tickers to a list
    frequent_tickers_list = frequent_tickers_df['Ticker'].tolist()

    # Return the list of frequent tickers
    return frequent_tickers_list

# %%
new_price_ticker = price_tickers(new_ticker)

# %%
hot_price_ticker= price_tickers(hot_ticker)

# %%
def download_stock_data(ticker):
    start_date = pd.to_datetime('2022-08-02')
    end_date = pd.to_datetime('2023-08-17')
    #download the stock
    data = yf.download(ticker, start=start_date, end=end_date)
    return data


# %%
def download_ticker(ticker_list,folder):
    for ticker in ticker_list:
        print(ticker)
        # Call the download_stock_data function to get adjusted closing prices
        stock_data = download_stock_data(ticker)
        #write the stock data to a csv file in the price_data folder
        stock_data.to_csv('price_data/'+folder+'/'+ticker+'.csv')

# %%
download_ticker(hot_price_ticker,'hot')

# %%
download_ticker(new_price_ticker,'new')

# %%
#test S&P 500 since the symbol in yf is unique
sp = download_stock_data("^GSPC")
sp.head()
#write it to a csv file
sp.to_csv('price_data/hot/S&P.csv')
sp.to_csv('price_data/new/S&P.csv')

# %%
def read_price_data(ticker_list,folder):
    data_dict = {}
    for ticker in ticker_list:
        file_path = os.path.join('price_data',f'{folder}' ,f'{ticker}.csv')
        if os.path.exists(file_path):
            data_dict[ticker] = pd.read_csv(file_path)
        else:
            print(f"File for {ticker} does not exist!")
    
    return data_dict

# %%
hot_price_data = read_price_data(hot_price_ticker,'hot')
new_price_data = read_price_data(new_price_ticker,'new')

# %%
hot_price_data['S&P'].head()

# %%
new_price_data['S&P'].head()

# %%
def calculate_returns(price_data):
    for ticker, df in price_data.items():
        close_prices = df['Close'].tolist()
        returns = [0]  # Initialize returns with 0 for the first item
        for i in range(1, len(close_prices)):
            ret = (close_prices[i] - close_prices[i-1]) / close_prices[i-1]
            returns.append(ret)
        df['Returns'] = returns
        price_data[ticker] = df  # Update the dictionary with the modified DataFrame
    return price_data


# %%
hot_price_data_ret = calculate_returns(hot_price_data)
hot_price_data_ret['S&P'].head()


# %%
new_price_data_ret = calculate_returns(new_price_data)
new_price_data_ret['AAPL'].head()

# %% [markdown]
# ## FEEDBACK - 1 Week calculation

# %%
def calculate_weekly_returns(price_data):
    for ticker, df in price_data.items():
        close_prices = df['Close']
        for days in range(1, 6):
            df[f'Returns_{days}'] = close_prices.pct_change(periods=days)
        price_data[ticker] = df
    return price_data

# %%
hot_price_data_weekly_ret = calculate_weekly_returns(hot_price_data)
hot_price_data_weekly_ret['S&P'].head(10)

# %%
new_price_data_weekly_ret = calculate_weekly_returns(new_price_data)
new_price_data_weekly_ret['S&P'].head(10)

# %% [markdown]
# # Add returns to the sentiment scores.

# %%
# read the sentiment files
hot_finbert = pd.read_csv('sentiment_scores/hot_finbert.csv')
hot_vader = pd.read_csv('sentiment_scores/hot_vader.csv')
hot_chatgpt = pd.read_csv('sentiment_scores/hot_chatgpt.csv')
hot_roberta = pd.read_csv('sentiment_scores/hot_roberta.csv')
new_finbert = pd.read_csv('sentiment_scores/new_finbert.csv')
new_vader = pd.read_csv('sentiment_scores/new_vader.csv')
new_chatgpt = pd.read_csv('sentiment_scores/new_chatgpt.csv')
new_roberta = pd.read_csv('sentiment_scores/new_roberta.csv')

# %%
def merge_returns(sentiment_df, price_data_ret):
    returns = []  # List to store the returns that will be matched

    # Iterate through the rows in sentiment_df
    for index, row in sentiment_df.iterrows():
        ticker = row['ticker']
        date = row['date']

        # Check if the ticker exists in price_data_ret
        if ticker in price_data_ret:
            ticker_df = price_data_ret[ticker]
        else:
            # If ticker not found, use the 'S&P' DataFrame
            ticker_df = price_data_ret['S&P']

        # Check if the date exists in the DataFrame for the ticker
        matching_date = ticker_df[ticker_df['Date'] == date]

        # If a matching date is found, get the corresponding return
        if not matching_date.empty:
            ret = matching_date['Returns'].iloc[0]
            returns.append(ret)
        else:
            returns.append(None)  # If date not found, append None

    # Add the returns list as a new column in sentiment_df
    sentiment_df['ret'] = returns

    return sentiment_df


# %%
hot_finbert_ret = merge_returns(hot_finbert,hot_price_data_ret)
print(hot_finbert_ret.head())
new_finbert_ret = merge_returns(new_finbert,new_price_data_ret)
print(new_finbert_ret.head())


# %%
hot_vader_ret = merge_returns(hot_vader,hot_price_data_ret)
print(hot_vader_ret.head())
new_vader_ret = merge_returns(new_vader,new_price_data_ret)
print(new_vader_ret.head())

# %%
hot_chatgpt_ret = merge_returns(hot_chatgpt,hot_price_data_ret)
print(hot_chatgpt_ret.head())
new_chatgpt_ret = merge_returns(new_chatgpt,new_price_data_ret)
print(new_chatgpt_ret.head())

# %%
hot_roberta_ret = merge_returns(hot_roberta,hot_price_data_ret)
print(hot_roberta_ret.head())
new_roberta_ret = merge_returns(new_roberta,new_price_data_ret)
print(new_roberta_ret.head())

# %%
hot_finbert_ret.to_csv('sentiment_score_ret/hot_finbert_ret.csv', index= False)
hot_vader_ret.to_csv('sentiment_score_ret/hot_vader_ret.csv', index= False)
hot_chatgpt_ret.to_csv('sentiment_score_ret/hot_chatgpt_ret.csv', index= False)
hot_roberta_ret.to_csv('sentiment_score_ret/hot_roberta_ret.csv', index=False)
new_finbert_ret.to_csv('sentiment_score_ret/new_finbert_ret.csv', index= False)
new_vader_ret.to_csv('sentiment_score_ret/new_vader_ret.csv', index= False)
new_chatgpt_ret.to_csv('sentiment_score_ret/new_chatgpt_ret.csv', index= False)
new_roberta_ret.to_csv('sentiment_score_ret/new_roberta_ret.csv', index=False)

# %%
hot_finbert_ret = pd.read_csv('sentiment_score_ret/hot_finbert_ret.csv')
hot_vader_ret = pd.read_csv('sentiment_score_ret/hot_vader_ret.csv')
hot_chatgpt_ret = pd.read_csv('sentiment_score_ret/hot_chatgpt_ret.csv')
hot_roberta_ret = pd.read_csv('sentiment_score_ret/hot_roberta_ret.csv')
new_finbert_ret = pd.read_csv('sentiment_score_ret/new_finbert_ret.csv')
new_vader_ret = pd.read_csv('sentiment_score_ret/new_vader_ret.csv')
new_chatgpt_ret = pd.read_csv('sentiment_score_ret/new_chatgpt_ret.csv')
new_roberta_ret = pd.read_csv('sentiment_score_ret/new_roberta_ret.csv')

# %%
def merge_weekly_returns(sentiment_df, price_data_ret):
    returns_columns = [f'Returns_{i}' for i in range(1, 6)]
    returns_dict = {col: [] for col in returns_columns}  # Dictionary to store the returns for each period

    # Iterate through the rows in sentiment_df
    for index, row in sentiment_df.iterrows():
        ticker = row['ticker']
        date = row['date']

        # Check if the ticker exists in price_data_ret
        ticker_df = price_data_ret.get(ticker, price_data_ret['S&P'])  # Use 'S&P' if ticker not found

        # Check if the date exists in the DataFrame for the ticker
        matching_date = ticker_df[ticker_df['Date'] == date]

        # If a matching date is found, get the corresponding returns for each period
        if not matching_date.empty:
            for col in returns_columns:
                returns_dict[col].append(matching_date[col].iloc[0])
        else:
            # If date not found, append None for each period
            for col in returns_columns:
                returns_dict[col].append(None)

    # Add the returns lists as new columns in sentiment_df
    for col, ret_values in returns_dict.items():
        sentiment_df[col] = ret_values

    return sentiment_df


# %%
#finbert
hot_finbert_weekly_ret = merge_weekly_returns(hot_finbert,hot_price_data_weekly_ret)
new_finbert_weekly_ret = merge_weekly_returns(new_finbert,new_price_data_weekly_ret)
#vader
hot_vader_weekly_ret = merge_weekly_returns(hot_vader,hot_price_data_weekly_ret)
new_vader_weekly_ret = merge_weekly_returns(new_vader,new_price_data_weekly_ret)
#chatgpt
hot_chatgpt_weekly_ret = merge_weekly_returns(hot_chatgpt,hot_price_data_weekly_ret)
new_chatgpt_weekly_ret = merge_weekly_returns(new_chatgpt,new_price_data_weekly_ret)
#roberta
hot_roberta_weekly_ret = merge_weekly_returns(hot_roberta,hot_price_data_weekly_ret)
new_roberta_weekly_ret = merge_weekly_returns(new_roberta,new_price_data_weekly_ret)
print("Done")

# %%
#Hot weekly return file
hot_finbert_weekly_ret.to_csv('sentiment_score_ret/weekly_data/hot_finbert_weekly_ret.csv', index= False)
hot_vader_weekly_ret.to_csv('sentiment_score_ret/weekly_data/hot_vader_weekly_ret.csv', index= False)
hot_chatgpt_weekly_ret.to_csv('sentiment_score_ret/weekly_data/hot_chatgpt_weekly_ret.csv', index= False)
hot_roberta_weekly_ret.to_csv('sentiment_score_ret/weekly_data/hot_roberta_weekly_ret.csv', index=False)
#New weekly return file
new_finbert_weekly_ret.to_csv('sentiment_score_ret/weekly_data/new_finbert_weekly_ret.csv', index= False)
new_vader_weekly_ret.to_csv('sentiment_score_ret/weekly_data/new_vader_weekly_ret.csv', index= False)
new_chatgpt_weekly_ret.to_csv('sentiment_score_ret/weekly_data/new_chatgpt_weekly_ret.csv', index= False)
new_roberta_weekly_ret.to_csv('sentiment_score_ret/weekly_data/new_roberta_weekly_ret.csv', index=False)

# %% [markdown]
# # Price Prediction
# 

# %%
import tensorflow as tf
from keras.layers import Input, Embedding, LSTM, Dense, Conv1D, MaxPooling1D, GlobalMaxPooling1D, concatenate
from keras.models import Model
from keras.utils import plot_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

# %%
def plot_loss_and_accuracy(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot training & validation loss values
    ax1.plot(history.history['loss'])
    ax1.plot(history.history['val_loss'])
    ax1.set_title('Model loss')
    ax1.set_ylabel('Loss')
    ax1.set_xlabel('Epoch')
    ax1.legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation accuracy values
    ax2.plot(history.history['accuracy'])
    ax2.plot(history.history['val_accuracy'])
    ax2.set_title('Model accuracy')
    ax2.set_ylabel('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.legend(['Train', 'Validation'], loc='upper left')

    plt.show()


# %%


# %% [markdown]
# # LSTM

# %%
def model_LSTM(df, max_words=10000):
    df['text'] = df['text'].astype(str)
    
    # Label encoding for 'label' column
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['label'])
    ticker_encoder = LabelEncoder()
    df['ticker'] = ticker_encoder.fit_transform(df['ticker'])

    df['score'] = df['score'].astype(float)
    # Separate features and target variable
    X = df[['text', 'label', 'ticker', 'score']]
    y = (df['ret'] > 0).astype(int)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Tokenize and pad the 'text' column for training and testing sets
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(X_train['text'])
    train_text_sequences = tokenizer.texts_to_sequences(X_train['text'])
    test_text_sequences = tokenizer.texts_to_sequences(X_test['text'])
    max_sequence_length = max(len(s) for s in train_text_sequences + test_text_sequences)
    train_text_padded = pad_sequences(train_text_sequences, maxlen=max_sequence_length)
    test_text_padded = pad_sequences(test_text_sequences, maxlen=max_sequence_length)

    # Text Input (LSTM)
    text_input = Input(shape=(max_sequence_length,), name='text_input')
    text_embedding = Embedding(input_dim=max_words, output_dim=128)(text_input)
    text_lstm = LSTM(128, return_sequences=True)(text_embedding)
    text_lstm2 = LSTM(64)(text_lstm) # Additional LSTM layer
    text_dense = Dense(32, activation='relu')(text_lstm2)

    # Ticker Input 
    ticker_input = Input(shape=(1,), name='ticker_input')
    ticker_dense = Dense(16, activation='relu')(ticker_input)

    # Label Input
    label_input = Input(shape=(1,), name='label_input')
    label_dense = Dense(16, activation='relu')(label_input)

    # Score Input
    score_input = Input(shape=(1,), name='score_input')
    score_dense1 = Dense(16, activation='relu')(score_input)
    score_dense2 = Dense(8, activation='relu')(score_dense1) # Additional dense layer

    # Concatenate all inputs
    concatenated = concatenate([text_dense, ticker_dense, label_dense, score_dense2])
    dense_1 = Dense(64, activation='relu')(concatenated)
    output = Dense(1, activation='sigmoid')(dense_1)

    # Build and compile the model
    model = Model(inputs=[text_input, ticker_input, label_input, score_input], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    #Print the model
    tf.keras.utils.plot_model(model, show_shapes=True)
    #model summary
    model.summary()

    # Train the model
    history = model.fit(
        {'text_input': train_text_padded, 'ticker_input': X_train['ticker'].values, 'label_input': X_train['label'].values, 'score_input': X_train['score'].values},
        y_train,
        epochs=7,
        batch_size=32,
        validation_split=0.2
    )

    # Plot the loss
    plot_loss_and_accuracy(history)

    # Evaluate the model
    loss, accuracy = model.evaluate(
        {'text_input': test_text_padded, 'ticker_input': X_test['ticker'].values, 'label_input': X_test['label'].values, 'score_input': X_test['score'].values},
        y_test
    )
    print("Test Accuracy: {:.2f}%".format(accuracy * 100))

# %%
print("=============  HOT FINBERT LSTM  =============== \n")
model_LSTM(hot_finbert_ret)
print("\n\n=============  NEW FINBERT LSTM  =============== ")
model_LSTM(new_finbert_ret)

# %%
print("=============  HOT CHAT-GPT 3.5 LSTM  =============== \n")
model_LSTM(hot_chatgpt_ret)
print("\n\n=============  NEW CHAT-GPT 3.5 LSTM  =============== ")
model_LSTM(new_chatgpt_ret)

# %%
def model_LSTM_vader(df, max_words=10000):
    df['text'] = df['text'].astype(str)

    # Label encoding for 'ticker' column
    ticker_encoder = LabelEncoder()
    df['ticker'] = ticker_encoder.fit_transform(df['ticker'])

    # Separate features and target variable
    X = df[['text', 'ticker', 'neg', 'neu', 'pos', 'compound']]
    y = (df['ret'] > 0).astype(int)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Tokenize and pad the 'text' column for training and testing sets
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(X_train['text'])
    train_text_sequences = tokenizer.texts_to_sequences(X_train['text'])
    test_text_sequences = tokenizer.texts_to_sequences(X_test['text'])
    max_sequence_length = max(len(s) for s in train_text_sequences + test_text_sequences)
    train_text_padded = pad_sequences(train_text_sequences, maxlen=max_sequence_length)
    test_text_padded = pad_sequences(test_text_sequences, maxlen=max_sequence_length)

    # Text Input (LSTM)
    text_input = Input(shape=(max_sequence_length,), name='text_input')
    text_embedding = Embedding(input_dim=max_words, output_dim=128)(text_input)
    text_lstm = LSTM(128, return_sequences=True)(text_embedding)
    text_lstm2 = LSTM(64)(text_lstm)
    text_dense = Dense(32, activation='relu')(text_lstm2)

    # Ticker Input (numeric)
    ticker_input = Input(shape=(1,), name='ticker_input')
    ticker_dense = Dense(16, activation='relu')(ticker_input)

    # Other Numeric Inputs (neg, neu, pos, compound)
    other_inputs = Input(shape=(4,), name='other_inputs')  # 4 numeric features
    other_dense = Dense(16, activation='relu')(other_inputs)

    # Concatenate all inputs
    concatenated = concatenate([text_dense, ticker_dense, other_dense])
    dense_1 = Dense(64, activation='relu')(concatenated)
    output = Dense(1, activation='sigmoid')(dense_1)

    # Build and compile the model
    model = Model(inputs=[text_input, ticker_input, other_inputs], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Model summary
    model.summary()

    # Train the model
    history = model.fit(
        {'text_input': train_text_padded, 'ticker_input': X_train['ticker'].values, 'other_inputs': X_train[['neg', 'neu', 'pos', 'compound']].values},
        y_train,
        epochs=7,
        batch_size=32,
        validation_split=0.2
    )

    # Plot the loss
    plot_loss_and_accuracy(history)

    # Evaluate the model
    loss, accuracy = model.evaluate(
        {'text_input': test_text_padded, 'ticker_input': X_test['ticker'].values, 'other_inputs': X_test[['neg', 'neu', 'pos', 'compound']].values},
        y_test
    )
    print("Test Accuracy: {:.2f}%".format(accuracy * 100))

    return model


# %%
print("=============  HOT CHAT-GPT 3.5 LSTM  =============== \n")
model_LSTM_vader(hot_vader_ret)
print("\n\n=============  NEW CHAT-GPT 3.5 LSTM  =============== ")
model_LSTM_vader(new_vader_ret)

# %%
print("=============  HOT ROBERTA LSTM  =============== \n")
model_LSTM(hot_roberta_ret)
print("\n\n=============  NEW ROBERTA LSTM  =============== ")
model_LSTM(new_roberta_ret)

# %% [markdown]
# # CNN

# %%
def model_CNN(df, max_words=10000):
    df['text'] = df['text'].astype(str)

    # One-hot encoding for 'label' and 'ticker' columns
    one_hot_encoder = OneHotEncoder(sparse=False)
    label_one_hot = one_hot_encoder.fit_transform(df[['label']])
    ticker_one_hot = one_hot_encoder.fit_transform(df[['ticker']])

    df['score'] = df['score'].astype(float)

    # Separate features and target variable
    X_text = df['text']
    X_ticker = ticker_one_hot
    X_label = label_one_hot
    X_score = df['score'].values.reshape(-1, 1)
    y = (df['ret'] > 0).astype(int)

    # Split the data into training and testing sets
    X_train_text, X_test_text, X_train_ticker, X_test_ticker, X_train_label, X_test_label, X_train_score, X_test_score, y_train, y_test = train_test_split(X_text, X_ticker, X_label, X_score, y, test_size=0.2, random_state=42)

    # Tokenize and pad the 'text' column for training and testing sets
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(X_train_text)
    train_text_sequences = tokenizer.texts_to_sequences(X_train_text)
    test_text_sequences = tokenizer.texts_to_sequences(X_test_text)
    max_sequence_length = max(len(s) for s in train_text_sequences + test_text_sequences)
    train_text_padded = pad_sequences(train_text_sequences, maxlen=max_sequence_length)
    test_text_padded = pad_sequences(test_text_sequences, maxlen=max_sequence_length)

    # Text Input (CNN)
    text_input = Input(shape=(max_sequence_length,), name='text_input')
    text_embedding = Embedding(input_dim=max_words, output_dim=128)(text_input)
    text_conv1 = Conv1D(128, 3, activation='relu')(text_embedding)  # Reduced kernel size
    text_pool1 = MaxPooling1D(3)(text_conv1)                        # Reduced pooling size
    text_conv2 = Conv1D(128, 3, activation='relu')(text_pool1)      # Reduced kernel size
    text_pool2 = MaxPooling1D(3)(text_conv2)                        # Reduced pooling size
    text_flat = GlobalMaxPooling1D()(text_pool2)
    text_dense = Dense(32, activation='relu')(text_flat)
    # Ticker Input (numeric)
    ticker_input_shape = X_ticker.shape[1]
    ticker_input = Input(shape=(ticker_input_shape,), name='ticker_input')

    # Label Input (numeric)
    label_input_shape = X_label.shape[1]
    label_input = Input(shape=(label_input_shape,), name='label_input')

    # Score Input (numeric)
    score_input = Input(shape=(1,), name='score_input')

    # Concatenate all inputs
    concatenated = concatenate([text_dense, ticker_input, label_input, score_input])
    dense_1 = Dense(64, activation='relu')(concatenated)
    output = Dense(1, activation='sigmoid')(dense_1)

    # Build and compile the model
    model = Model(inputs=[text_input, ticker_input, label_input, score_input], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Model summary
    model.summary()

    # Train the model
    history = model.fit(
        {'text_input': train_text_padded, 'ticker_input': X_train_ticker, 'label_input': X_train_label, 'score_input': X_train_score},
        y_train,
        epochs=7,
        batch_size=32,
        validation_split=0.2
    )

    # Plot the loss
    plot_loss_and_accuracy(history)

    # Evaluate the model
    loss, accuracy = model.evaluate(
        {'text_input': test_text_padded, 'ticker_input': X_test_ticker, 'label_input': X_test_label, 'score_input': X_test_score},
        y_test
    )
    print("Test Accuracy: {:.2f}%".format(accuracy * 100))

# %%
print("=============  HOT FINBERT CNN  =============== \n")
model_CNN(hot_finbert_ret)
print("\n\n=============  NEW FINERT CNN  =============== ")
model_CNN(new_finbert_ret)

# %%
print("=============  HOT CHAT GPT 3.5 CNN  =============== \n")
model_CNN(hot_chatgpt_ret)
print("\n\n=============  NEW CHAT GPT 3.5 CNN  =============== ")
model_CNN(new_chatgpt_ret)

# %%
def model_CNN_vader(df, max_words=10000):
    df['text'] = df['text'].astype(str)

    # One-hot encoding for 'ticker' column
    one_hot_encoder = OneHotEncoder(sparse=False)
    ticker_one_hot = one_hot_encoder.fit_transform(df[['ticker']])

    # Separate features and target variable
    X_text = df['text']
    X_ticker = ticker_one_hot
    X_neg = df['neg'].values.reshape(-1, 1)
    X_neu = df['neu'].values.reshape(-1, 1)
    X_pos = df['pos'].values.reshape(-1, 1)
    X_compound = df['compound'].values.reshape(-1, 1)
    y = (df['ret'] > 0).astype(int)

    # Split the data into training and testing sets
    splits = train_test_split(X_text, X_ticker, X_neg, X_neu, X_pos, X_compound, y, test_size=0.2, random_state=42)
    X_train_text, X_test_text, X_train_ticker, X_test_ticker, X_train_neg, X_test_neg, X_train_neu, X_test_neu, X_train_pos, X_test_pos, X_train_compound, X_test_compound, y_train, y_test = splits

    # Tokenize and pad the 'text' column
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(X_train_text)
    train_text_sequences = tokenizer.texts_to_sequences(X_train_text)
    test_text_sequences = tokenizer.texts_to_sequences(X_test_text)
    max_sequence_length = max(len(s) for s in train_text_sequences + test_text_sequences)
    train_text_padded = pad_sequences(train_text_sequences, maxlen=max_sequence_length)
    test_text_padded = pad_sequences(test_text_sequences, maxlen=max_sequence_length)

    # Text Input (CNN)
    text_input = Input(shape=(max_sequence_length,), name='text_input')
    text_embedding = Embedding(input_dim=max_words, output_dim=128)(text_input)
    text_conv1 = Conv1D(128, 3, activation='relu')(text_embedding)
    text_pool1 = MaxPooling1D(3)(text_conv1)
    text_conv2 = Conv1D(128, 3, activation='relu')(text_pool1)
    text_pool2 = MaxPooling1D(3)(text_conv2)
    text_flat = GlobalMaxPooling1D()(text_pool2)
    text_dense = Dense(32, activation='relu')(text_flat)

    # Additional Inputs (numeric)
    ticker_input = Input(shape=(X_ticker.shape[1],), name='ticker_input')
    neg_input = Input(shape=(1,), name='neg_input')
    neu_input = Input(shape=(1,), name='neu_input')
    pos_input = Input(shape=(1,), name='pos_input')
    compound_input = Input(shape=(1,), name='compound_input')

    # Concatenate all inputs
    concatenated = concatenate([text_dense, ticker_input, neg_input, neu_input, pos_input, compound_input])
    dense_1 = Dense(64, activation='relu')(concatenated)
    output = Dense(1, activation='sigmoid')(dense_1)

    # Build and compile the model
    model = Model(inputs=[text_input, ticker_input, neg_input, neu_input, pos_input, compound_input], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(
        {'text_input': train_text_padded, 'ticker_input': X_train_ticker, 'neg_input': X_train_neg, 'neu_input': X_train_neu, 'pos_input': X_train_pos, 'compound_input': X_train_compound},
        y_train,
        epochs=7,
        batch_size=32,
        validation_split=0.2
    )

    # Plot the loss
    plot_loss_and_accuracy(history)

    # Evaluate the model
    loss, accuracy = model.evaluate(
        {'text_input': test_text_padded, 'ticker_input': X_test_ticker, 'neg_input': X_test_neg, 'neu_input': X_test_neu, 'pos_input': X_test_pos, 'compound_input': X_test_compound},
        y_test
    )
    print("Test Accuracy: {:.2f}%".format(accuracy * 100))

# %%
print("=============  HOT VADER CNN  =============== \n")
model_CNN_vader(hot_vader_ret)
print("\n\n=============  NEW VADER CNN  =============== ")
model_CNN_vader(new_vader_ret)

# %%
print("=============  HOT ROBERTA CNN  =============== \n")
model_CNN(hot_roberta_ret)
print("\n\n=============  NEW ROBERTA CNN  =============== ")
model_CNN(new_roberta_ret)

# %% [markdown]
# # CNN-LSTM

# %%
def model_CNN_LSTM(df, max_words=10000):
    df['text'] = df['text'].astype(str)

    # One-hot encoding for 'label' and 'ticker' columns
    one_hot_encoder = OneHotEncoder(sparse=False)
    label_one_hot = one_hot_encoder.fit_transform(df[['label']])
    ticker_one_hot = one_hot_encoder.fit_transform(df[['ticker']])
    df['score'] = df['score'].astype(float)

    # Separate features and target variable
    X_text = df['text']
    X_ticker = ticker_one_hot
    X_label = label_one_hot
    X_score = df['score'].values.reshape(-1, 1)
    y = (df['ret'] > 0).astype(int)

    # Split the data into training and testing sets
    X_train_text, X_test_text, X_train_ticker, X_test_ticker, X_train_label, X_test_label, X_train_score, X_test_score, y_train, y_test = train_test_split(X_text, X_ticker, X_label, X_score, y, test_size=0.2, random_state=42)

    # Tokenize and pad the 'text' column for training and testing sets
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(X_train_text)
    train_text_sequences = tokenizer.texts_to_sequences(X_train_text)
    test_text_sequences = tokenizer.texts_to_sequences(X_test_text)
    max_sequence_length = max(len(s) for s in train_text_sequences + test_text_sequences)
    train_text_padded = pad_sequences(train_text_sequences, maxlen=max_sequence_length)
    test_text_padded = pad_sequences(test_text_sequences, maxlen=max_sequence_length)

    # Text Input (CNN followed by LSTM)
    text_input = Input(shape=(max_sequence_length,), name='text_input')
    text_embedding = Embedding(input_dim=max_words, output_dim=128)(text_input)
    text_conv1 = Conv1D(128, 3, activation='relu')(text_embedding)
    text_pool1 = MaxPooling1D(3)(text_conv1)
    text_conv2 = Conv1D(128, 3, activation='relu')(text_pool1)
    text_pool2 = MaxPooling1D(3)(text_conv2)
    text_lstm1 = LSTM(128, return_sequences=True)(text_pool2)
    text_lstm2 = LSTM(64)(text_lstm1)  # Additional LSTM layer
    text_dense = Dense(32, activation='relu')(text_lstm2)

    # Ticker Input (numeric)
    ticker_input_shape = X_ticker.shape[1]
    ticker_input = Input(shape=(ticker_input_shape,), name='ticker_input')
    ticker_dense = Dense(16, activation='relu')(ticker_input)

    # Label Input (numeric)
    label_input_shape = X_label.shape[1]
    label_input = Input(shape=(label_input_shape,), name='label_input')
    label_dense = Dense(16, activation='relu')(label_input)

    # Score Input (numeric)
    score_input = Input(shape=(1,), name='score_input')
    score_dense1 = Dense(16, activation='relu')(score_input)
    score_dense2 = Dense(8, activation='relu')(score_dense1)  # Additional dense layer

    # Concatenate all inputs
    concatenated = concatenate([text_dense, ticker_dense, label_dense, score_dense2])
    dense_1 = Dense(64, activation='relu')(concatenated)
    output = Dense(1, activation='sigmoid')(dense_1)

    # Build and compile the model
    model = Model(inputs=[text_input, ticker_input, label_input, score_input], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Model summary
    model.summary()

    # Train the model
    history = model.fit(
        {'text_input': train_text_padded, 'ticker_input': X_train_ticker, 'label_input': X_train_label, 'score_input': X_train_score},
        y_train,
        epochs=7,
        batch_size=32,
        validation_split=0.2
    )

    # Plot the loss
    plot_loss_and_accuracy(history)

    # Evaluate the model
    loss, accuracy = model.evaluate(
        {'text_input': test_text_padded, 'ticker_input': X_test_ticker, 'label_input': X_test_label, 'score_input': X_test_score},
        y_test
    )
    print("Test Accuracy: {:.2f}%".format(accuracy * 100))


# %%
print("=============  HOT FINBERT CNN-LSTM  =============== \n")
model_CNN_LSTM(hot_finbert_ret)
print("\n\n=============  NEW FINERT CNN-LSTM   =============== ")
model_CNN_LSTM(new_finbert_ret)

# %%
print("=============  HOT CHAT GPT 3.5 CNN-LSTM  =============== \n")
model_CNN_LSTM(hot_chatgpt_ret)
print("\n\n=============  NEW CHAT GPT 3.5 CNN-LSTM   =============== ")
model_CNN_LSTM(new_chatgpt_ret)

# %%
def model_CNN_LSTM_vader(df, max_words=10000):
    df['text'] = df['text'].astype(str)

    # One-hot encoding for 'ticker' column
    one_hot_encoder = OneHotEncoder(sparse=False)
    ticker_one_hot = one_hot_encoder.fit_transform(df[['ticker']])
    
    # Separate features and target variable
    X_text = df['text']
    X_ticker = ticker_one_hot
    X_neg = df['neg'].values.reshape(-1, 1)
    X_neu = df['neu'].values.reshape(-1, 1)
    X_pos = df['pos'].values.reshape(-1, 1)
    X_compound = df['compound'].values.reshape(-1, 1)
    y = (df['ret'] > 0).astype(int)

    # Split the data into training and testing sets
    X_train_text, X_test_text, X_train_ticker, X_test_ticker, X_train_neg, X_test_neg, X_train_neu, X_test_neu, X_train_pos, X_test_pos, X_train_compound, X_test_compound, y_train, y_test = train_test_split(X_text, X_ticker, X_neg, X_neu, X_pos, X_compound, y, test_size=0.2, random_state=42)

    # Tokenize and pad the 'text' column for training and testing sets
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(X_train_text)
    train_text_sequences = tokenizer.texts_to_sequences(X_train_text)
    test_text_sequences = tokenizer.texts_to_sequences(X_test_text)
    max_sequence_length = max(len(s) for s in train_text_sequences + test_text_sequences)
    train_text_padded = pad_sequences(train_text_sequences, maxlen=max_sequence_length)
    test_text_padded = pad_sequences(test_text_sequences, maxlen=max_sequence_length)

    # Text Input (CNN followed by LSTM)
    text_input = Input(shape=(max_sequence_length,), name='text_input')
    text_embedding = Embedding(input_dim=max_words, output_dim=128)(text_input)
    text_conv1 = Conv1D(128, 3, activation='relu')(text_embedding)
    text_pool1 = MaxPooling1D(3)(text_conv1)
    text_conv2 = Conv1D(128, 3, activation='relu')(text_pool1)
    text_pool2 = MaxPooling1D(3)(text_conv2)
    text_lstm1 = LSTM(128, return_sequences=True)(text_pool2)
    text_lstm2 = LSTM(64)(text_lstm1)  # Additional LSTM layer
    text_dense = Dense(32, activation='relu')(text_lstm2)

    # Ticker Input (numeric)
    ticker_input_shape = X_ticker.shape[1]
    ticker_input = Input(shape=(ticker_input_shape,), name='ticker_input')
    ticker_dense = Dense(16, activation='relu')(ticker_input)

    # Sentiment Scores Input (numeric)
    neg_input = Input(shape=(1,), name='neg_input')
    neu_input = Input(shape=(1,), name='neu_input')
    pos_input = Input(shape=(1,), name='pos_input')
    compound_input = Input(shape=(1,), name='compound_input')

    # Concatenate sentiment scores
    sentiment_scores = concatenate([neg_input, neu_input, pos_input, compound_input])
    sentiment_dense = Dense(16, activation='relu')(sentiment_scores)

    # Concatenate all inputs
    concatenated = concatenate([text_dense, ticker_dense, sentiment_dense])
    dense_1 = Dense(64, activation='relu')(concatenated)
    output = Dense(1, activation='sigmoid')(dense_1)

    # Build and compile the model
    model = Model(inputs=[text_input, ticker_input, neg_input, neu_input, pos_input, compound_input], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Model summary
    model.summary()

    # Train the model
    history = model.fit(
        {'text_input': train_text_padded, 'ticker_input': X_train_ticker, 'neg_input': X_train_neg, 'neu_input': X_train_neu, 'pos_input': X_train_pos, 'compound_input': X_train_compound},
        y_train,
        epochs=7,
        batch_size=32,
        validation_split=0.2
    )

    # Plot the loss (you'll need to define or import a function to do this)
    plot_loss_and_accuracy(history)

    # Evaluate the model
    loss, accuracy = model.evaluate(
        {'text_input': test_text_padded, 'ticker_input': X_test_ticker, 'neg_input': X_test_neg, 'neu_input': X_test_neu, 'pos_input': X_test_pos, 'compound_input': X_test_compound},
        y_test
    )
    print("Test Accuracy: {:.2f}%".format(accuracy * 100))


# %%
print("=============  HOT VADER CNN-LSTM  =============== \n")
model_CNN_LSTM_vader(hot_vader_ret)
print("\n\n=============  NEW VADER CNN-LSTM   =============== ")
model_CNN_LSTM_vader(new_vader_ret)

# %%
print("=============  HOT ROBERTA CNN_LSTM  =============== \n")
model_CNN_LSTM(hot_roberta_ret)
print("\n\n=============  NEW ROBERTA CNN_LSTM  =============== ")
model_CNN_LSTM(new_roberta_ret)

# %% [markdown]
# # Weekly Price Prediction

# %%
def model_LSTM_weekly(df, return_day=1, max_words=10000):
    df['text'] = df['text'].astype(str)
    
    # Label encoding for 'label' column
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['label'])
    ticker_encoder = LabelEncoder()
    df['ticker'] = ticker_encoder.fit_transform(df['ticker'])

    df['score'] = df['score'].astype(float)
    # Separate features and target variable
    X = df[['text', 'label', 'ticker', 'score']]
    y = (df[f'Returns_{return_day}'] > 0).astype(int)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Tokenize and pad the 'text' column for training and testing sets
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(X_train['text'])
    train_text_sequences = tokenizer.texts_to_sequences(X_train['text'])
    test_text_sequences = tokenizer.texts_to_sequences(X_test['text'])
    max_sequence_length = max(len(s) for s in train_text_sequences + test_text_sequences)
    train_text_padded = pad_sequences(train_text_sequences, maxlen=max_sequence_length)
    test_text_padded = pad_sequences(test_text_sequences, maxlen=max_sequence_length)

    # Text Input (LSTM)
    text_input = Input(shape=(max_sequence_length,), name='text_input')
    text_embedding = Embedding(input_dim=max_words, output_dim=128)(text_input)
    text_lstm = LSTM(128, return_sequences=True)(text_embedding)
    text_lstm2 = LSTM(64)(text_lstm) # Additional LSTM layer
    text_dense = Dense(32, activation='relu')(text_lstm2)

    # Ticker Input 
    ticker_input = Input(shape=(1,), name='ticker_input')
    ticker_dense = Dense(16, activation='relu')(ticker_input)

    # Label Input
    label_input = Input(shape=(1,), name='label_input')
    label_dense = Dense(16, activation='relu')(label_input)

    # Score Input
    score_input = Input(shape=(1,), name='score_input')
    score_dense1 = Dense(16, activation='relu')(score_input)
    score_dense2 = Dense(8, activation='relu')(score_dense1) # Additional dense layer

    # Concatenate all inputs
    concatenated = concatenate([text_dense, ticker_dense, label_dense, score_dense2])
    dense_1 = Dense(64, activation='relu')(concatenated)
    output = Dense(1, activation='sigmoid')(dense_1)

    # Build and compile the model
    model = Model(inputs=[text_input, ticker_input, label_input, score_input], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    #Print the model
    tf.keras.utils.plot_model(model, show_shapes=True)
    #model summary
    #model.summary()

    # Train the model
    history = model.fit(
        {'text_input': train_text_padded, 'ticker_input': X_train['ticker'].values, 'label_input': X_train['label'].values, 'score_input': X_train['score'].values},
        y_train,
        epochs=5,
        batch_size=32,
        validation_split=0.2
    )

    # Plot the loss
    plot_loss_and_accuracy(history)

    # Evaluate the model
    loss, accuracy = model.evaluate(
        {'text_input': test_text_padded, 'ticker_input': X_test['ticker'].values, 'label_input': X_test['label'].values, 'score_input': X_test['score'].values},
        y_test
    )
    print("Test Accuracy for Returns from day ",return_day," : {:.2f}%".format(accuracy * 100))

# %% [markdown]
# # FINBERT WEEKLY

# %%
print("=============  HOT FINBERT LSTM  =============== \n")
for i in range(1,6):
    model_LSTM_weekly(hot_finbert_weekly_ret,i)

# %% [markdown]
# ## NEW FINBERT WEEKLY

# %%
print("=============  NEW FINBERT LSTM  =============== \n")
for i in range(1,6):
    model_LSTM_weekly(new_finbert_weekly_ret,i)

# %% [markdown]
# # CHATGPT WEEKLY

# %%
print("=============  HOT CHAT GPT 3.5 LSTM  =============== \n")
for i in range(1,6):
    model_LSTM_weekly(hot_chatgpt_weekly_ret,i)

# %% [markdown]
# ## NEW CHATGPT

# %%
print("=============  NEW CHAT GPT 3.5 LSTM  =============== \n")
for i in range(1,6):
    model_LSTM_weekly(new_chatgpt_weekly_ret,i)

# %% [markdown]
# # ROBERTA WEEKLY

# %%
print("=============  HOT ROBERTA LSTM  =============== \n")
for i in range(1,6):
    model_LSTM_weekly(hot_roberta_weekly_ret,i)


