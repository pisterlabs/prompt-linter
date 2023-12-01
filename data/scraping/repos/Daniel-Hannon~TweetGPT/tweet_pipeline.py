from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time
import datetime
import sqlite3

from langchain.llms import OpenAI
import sqlite3
from typing import List

from selenuim_test import get_trend_tweets, get_trends
from generate_new_tweet import get_tweets, get_best_tweets, generate_new_tweet

your_open_ai_key = ""

model_name = "gpt-3.5-turbo-instruct"

llm = OpenAI(openai_api_key=your_open_ai_key, model_name=model_name, max_tokens=100)


con = sqlite3.connect("twitter.db")
cur = con.cursor()

chrome_options = Options()
#Put your own path to the chrome user data folder
chrome_options.add_argument(r"--user-data-dir=C:\Users\danie\AppData\Local\Google\Chrome\User Data")

chrome_options.add_argument("--headless")
# chrome_options.add_argument("--full-screen")

if __name__ == "__main__":
   driver, trend_names = get_trends(chrome_options)
   new_tweets = []
   for trend in trend_names:
        promoted= get_trend_tweets(driver, trend)
        # get the tweets for the trend
        if not promoted:
            tweets = get_tweets(trend)
            # loop through the tweets
            best_tweets = get_best_tweets(tweets)
            #generate a new tweet
            new_tweet = generate_new_tweet(trend, best_tweets, llm)
            print("Trend is: " + trend)
            print("New tweet is: " + new_tweet)
            #add the new tweet to the list of new tweets
            new_tweets.append(new_tweet)
    
    #tweet the new tweets  TODO:: Figure out how to tweet using twitter api
    # for tweet in new_tweets:
    #     tweet_new_tweet(tweet)

   con.close()

