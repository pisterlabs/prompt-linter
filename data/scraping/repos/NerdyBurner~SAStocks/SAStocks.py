# Importing required tools
import pandas as pd
from pandas.tseries.offsets import BDay
import requests
from datetime import datetime, timedelta
import openai
from sqlite3 import dbapi2 as sqlite
from retrying import retry
import time
from datetime import datetime
import traceback
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('vader_lexicon')
import pickle
import os
import logging

def save_state(processed_tickers, report_data):
    state = {"processed_tickers": processed_tickers, "report_data": report_data}
    with open("state.pkl", "wb") as f:
        pickle.dump(state, f)

def load_state():
    if os.path.exists("state.pkl"):
        with open("state.pkl", "rb") as f:
            state = pickle.load(f)
        return state["processed_tickers"], state["report_data"]
    else:
        return [], []

# Configure logging
logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)

# Replace print statements with logging statements
logging.info("This is an info message")
logging.error("This is an error message")

# Load API keys from CSV
keys_df = pd.read_csv('api_keys.csv')
openai_key = keys_df['openai_key'].values[0]
polygon_key = keys_df['polygon_key'].values[0]

# OpenAI and Polygon.io API setup
openai.api_key = openai_key
polygon_url = "https://api.polygon.io/v1/meta/symbols/{ticker}/news?perpage=50&page=1&apiKey=" + polygon_key

def load_tickers():
    df = pd.read_csv('Tickers.csv')
    tickers = df.iloc[:, 2].tolist()
    return tickers

# Database connection for news articles
news_database_filename = 'news_articles.db'
news_connection = sqlite.connect(news_database_filename)

news_connection.cursor().execute('''
    CREATE TABLE IF NOT EXISTS news_articles
    (date text, ticker text, title text, description text)
''')

def save_news_to_db(date, ticker, title, description):
    query = """
        INSERT INTO news_articles 
        (date, ticker, title, description) 
        VALUES (?, ?, ?, ?)
    """
    news_connection.cursor().execute(query, (date, ticker, title, description))
    news_connection.commit()

# Database connection
database_filename = 'sentiment_scores.db'
connection = sqlite.connect(database_filename)

connection.cursor().execute('''
    CREATE TABLE IF NOT EXISTS sentiment_scores
    (date text, ticker text, vader_sentiment text, gpt_sentiment text, gpt_response text,
        historical_price_high real, historical_price_low real, historical_price_avg real, aggregated_score real,
        recent_price real, rsi real, macd real)
''')

# Vader Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

def get_latest_date_in_db():
    cursor = connection.cursor()
    cursor.execute("SELECT MAX(date) FROM sentiment_scores")
    result = cursor.fetchone()
    if result[0] is not None:
        return datetime.strptime(result[0], '%Y-%m-%d').date()
    else:
        return None

@retry(stop_max_attempt_number=7, wait_exponential_multiplier=1000, wait_exponential_max=10000)
def requests_get_with_retry(url):
    response = requests.get(url)
    response.raise_for_status()  # Raises stored HTTPError, if one occurred
    return response

def get_stock_news(ticker):
    try:
        url = f"https://api.polygon.io/v2/reference/news?ticker={ticker}&apiKey={polygon_key}"
        response = requests.get(url)
        data = response.json()

        if 'status' in data and data['status'] == "OK":
            for result in data['results']:
                timestamp = result.get('timestamp', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                save_news_to_db(timestamp, ticker, result.get('title', ''), result.get('description', ''))
            return data['results']
        else:
            return []
    except Exception as e:
        print(f"Error getting stock news for {ticker}: {e}")
        return []

# Initialize a global dictionary to hold all articles
ticker_articles = {}

def add_article_to_tickers(article, ticker_articles):
    # Create a dictionary for the article
    article_dict = {
        'title': article.get('title', ''),
        'description': article.get('description', ''),
        'keywords': article.get('keywords', []),  # Use the .get method to prevent KeyError
    }

    # Loop over each ticker
    for ticker in article['tickers']:
        # If the ticker is not already in the dictionary, add it
        if ticker not in ticker_articles:
            ticker_articles[ticker] = []
        
        # Add the article to the list of articles for this ticker
        ticker_articles[ticker].append(article_dict)
    
    return ticker_articles

def get_news_from_db(ticker):
    cursor = news_connection.cursor()
    cursor.execute("SELECT date, title, description FROM news_articles WHERE ticker=?", (ticker,))
    results = cursor.fetchall()
    return results

def vader_sentiment_analysis(ticker):
    news_articles = get_news_from_db(ticker)
    sentiment_scores = []
    for date, title, description in news_articles:
        if title and description:
            text = title + " " + description
            sentiment_score = analyzer.polarity_scores(text)
            compound_score = sentiment_score['compound']
            if compound_score >= 0.05:
                sentiment_scores.append("Good")
            elif compound_score <= -0.05:
                sentiment_scores.append("Bad")
            else:
                sentiment_scores.append("Neutral")
    return sentiment_scores

def gpt_sentiment_analysis(ticker, max_retries=4, retry_delay=2.0):
    news_articles = get_news_from_db(ticker)
    sentiment_scores = []
    for date, title, description in news_articles:
        if title and description:
            text = title + " " + description
            for i in range(max_retries):
                try:
                    prompt = f"As an analyst, assess the sentiment of the following information: {text}. Would you categorize it as 'Good', 'Bad', or 'Unknown' in the context of {ticker}?"
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are an analyst whose task is to assess the sentiment of financial news."},
                            {"role": "user", "content": prompt},
                        ],
                        max_tokens=300)
                        
                    full_response = response['choices'][0]['message']['content'].strip()
                    sentiment = full_response.split('\n')[0].strip().upper()

                    if sentiment not in ['Good', 'Bad', 'Neutral']:
                        sentiment = 'UNKNOWN'

                    sentiment_scores.append(sentiment)
                except Exception as e:
                    print(f"Error during GPT sentiment analysis: {e}")
                    if i < max_retries - 1:  # if not the last retry attempt
                        time.sleep(retry_delay)  # wait before next retry
                    else:
                        sentiment_scores.append('Error')  # return Error after all retry attempts failed
    return sentiment_scores

def get_rsi(ticker, polygon_key):
    rsi_url = f"https://api.polygon.io/v1/indicators/rsi/{ticker}?apiKey={polygon_key}"
    rsi_response = requests_get_with_retry(rsi_url)
    if rsi_response.status_code == 200:
        rsi_data = rsi_response.json()['results']['values']
        return rsi_data[0]['value'] if rsi_data else None  # Return the most recent value
    else:
        print(f"Error in get_rsi() for ticker {ticker}: {rsi_response.status_code}")
        return None

def get_macd(ticker, polygon_key):
    macd_url = f"https://api.polygon.io/v1/indicators/macd/{ticker}?apiKey={polygon_key}"
    macd_response = requests_get_with_retry(macd_url)
    if macd_response.status_code == 200:
        macd_data = macd_response.json()['results']['values']
        return macd_data[0]['value'] if macd_data else None  # Return the most recent value
    else:
        print(f"Error in get_macd() for ticker {ticker}: {macd_response.status_code}")
        return None

def get_historical_price(ticker):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    response = requests_get_with_retry(
        f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}?apiKey={polygon_key}"
    )
    data = response.json()["results"]
    prices = [item["c"] for item in data]  # closing prices
    if prices:
        return max(prices), min(prices), sum(prices) / len(prices)
    else:
        return None, None, None

def get_recent_price(ticker):
    current_date = (datetime.now() - BDay(1)).strftime("%Y-%m-%d")  # Get the most recent business day
    response = requests_get_with_retry(f"https://api.polygon.io/v1/open-close/{ticker}/{current_date}?adjusted=true&apiKey={polygon_key}")
    data = response.json() if response else None
    if data and 'close' in data:
        recent_price = data['close']
    else:
        recent_price = None
    return recent_price

def calculate_aggregated_score(vader_total, gpt_total, historical_price_low, historical_price_high, recent_price, news_volume, rsi, macd, num_articles):
    vader_score = vader_total / num_articles
    gpt_score = gpt_total / num_articles
    
    price_score = 0
    if recent_price < historical_price_low:
        price_score = -1
    elif recent_price > historical_price_high:
        price_score = 1

    news_volume_score = 0
    if news_volume > 10:
        news_volume_score = 1
    elif news_volume < 5:
        news_volume_score = -1

    rsi_score = 0
    if rsi < 30:
        rsi_score = -1
    elif rsi > 70:
        rsi_score = 1

    macd_score = 0
    if macd < 0:
        macd_score = -1
    elif macd > 0:
        macd_score = 1

    aggregated_score = (vader_score + gpt_score + price_score + news_volume_score + rsi_score + macd_score) / 6

    return aggregated_score

def save_to_db(date, ticker, vader_sentiment, gpt_sentiment, gpt_response, historical_price_high, historical_price_low, historical_price_avg, aggregated_score, recent_price, rsi, macd):
    query = """
        INSERT INTO sentiment_scores 
        (date, ticker, vader_sentiment, gpt_sentiment, gpt_response, historical_price_high, historical_price_low, historical_price_avg, aggregated_score, recent_price, rsi, macd) 
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    connection.cursor().execute(query, (date, ticker, vader_sentiment, gpt_sentiment, gpt_response, historical_price_high, historical_price_low, historical_price_avg, aggregated_score, recent_price, rsi, macd))
    connection.commit()  # commit right after inserting data for a ticker

def print_report(report_data):
    print("Here is your report:")
    for ticker, score in report_data:
        print(f"{ticker}: {score}")
    # Write the report to a file
    with open('report.txt', 'w') as f:
        f.write("Report:\n")
        for ticker, score in report_data:
            f.write(f"{ticker}: {score}\n")
    print("Report written to report.txt")

def main():
    sentiment_scores = {"Good": 1, "Neutral": 0, "UNKNOWN": 0, "ERROR": 0, "Error": 0, "Bad": -1}

    try:
        # Load state
        try:
            with open("state.pkl", "rb") as f:
                state = pickle.load(f)
            processed_tickers = state["processed_tickers"]
            report_data = state["report_data"]
        except FileNotFoundError:
            processed_tickers = []
            report_data = []

        # Load tickers
        print("Starting Analysis - Reading Tickers.csv")
        tickers = load_tickers()
        print(f"Loaded {len(tickers)} tickers.")
        if not tickers:
            print("No tickers found. Exiting.")
            return
        
        report_data = []

        # Initialize dictionaries to store news, RSI, and MACD data for each ticker
        news_data = {}
        rsi_data = {}
        macd_data = {}

        # Download news articles for all tickers
        print("Importing and Filtering News from Polygon.io for all tickers")
        for i, ticker in enumerate(tickers, start=1):
            if ticker in processed_tickers:
                continue  # Skip tickers that have been processed already
            print(f"\nImporting news for ticker #{i}: {ticker}")
            news = get_stock_news(ticker)
            print(f"Loaded and saved {len(news)} news articles for {ticker} in the database.")
            print(f"Finished importing and filtering news. Total articles: {len(ticker_articles)}")

            # Save the news articles for this ticker
            news_data[ticker] = news

            # Get RSI and MACD
            rsi = get_rsi(ticker, polygon_key)
            macd = get_macd(ticker, polygon_key)

            # Save the RSI and MACD data for this ticker
            rsi_data[ticker] = rsi
            macd_data[ticker] = macd

        # Analyze news articles for all tickers
        print("Conducting sentiment analysis and scoring")
        for i, ticker in enumerate(tickers, start=1):
            print(f"Processing ticker #{i}: {ticker}")
            # Skip tickers that have been processed already
            if ticker in processed_tickers:
                continue

            # Get the news, RSI, and MACD data for this ticker
            news = news_data[ticker]
            rsi = rsi_data[ticker]
            macd = macd_data[ticker]

            vader_sentiments = vader_sentiment_analysis(ticker)
            gpt_sentiments = gpt_sentiment_analysis(ticker)
            # Skip ticker if no articles were processed
            if len(vader_sentiments) == 0 or len(gpt_sentiments) == 0:
                print(f"No articles were processed for {ticker}. Skipping.")
                continue

            # Get recent price
            recent_price = get_recent_price(ticker)

            # Calculate scores and save to database
            historical_price_high, historical_price_low, historical_price_avg = get_historical_price(ticker)
            # Now that all articles have been processed, calculate the aggregated score
            vader_total = sum([sentiment_scores[sentiment] for sentiment in vader_sentiments])
            gpt_total = sum([sentiment_scores[sentiment] for sentiment in gpt_sentiments])
            aggregated_score = calculate_aggregated_score(vader_total, gpt_total, historical_price_low, historical_price_high, recent_price, len(news), rsi, macd, len(vader_sentiments))

            # Print calculated final scoring
            print(f"Aggregated Score for {ticker}: {aggregated_score}")

            # Add the ticker and score to the report data
            report_data.append((ticker, aggregated_score))

            print(f"Finished Processing Ticker #{i}: {ticker}")

            # After processing each ticker, add it to the list of processed tickers and save the state
            processed_tickers.append(ticker)
            state = {"processed_tickers": processed_tickers, "report_data": report_data}
            with open("state.pkl", "wb") as f:
                pickle.dump(state, f)

        # Sort the report data by score in descending order
        report_data.sort(key=lambda x: x[1], reverse=True)

        # Print and write the report
        print_report(report_data)

    except Exception as e:
        print(f"Error processing ticker {ticker}: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()

