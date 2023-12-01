from langchain.document_loaders import UnstructuredHTMLLoader
from bs4 import BeautifulSoup
import re
import pandas as pd
from tqdm.notebook import tqdm
import snscrape.modules.twitter as sntwitter
import logging
import os
from dotenv import load_dotenv

def twitter(twitter_handle):
    print(f'Searching for {twitter_handle}')
    try:
        # Set up logging to a file with UTF-8 encoding
        # log_file = 'logs/twitter.log'
        # logging.basicConfig(filename=log_file, encoding='utf-8', level=logging.INFO)

        # Scrape tweets using sntwitter
        scraper = sntwitter.TwitterSearchScraper(twitter_handle)

        tweets = []
        for i, tweet in enumerate(scraper.get_items()):
            data = [
                tweet.date,
                #tweet.id,
                tweet.content,
                tweet.user.username,
                tweet.likeCount,
                #tweet.retweetCount,
            ]
            tweets.append(data)
            if i > 50:
                break

        tweet_df = pd.DataFrame(tweets, columns=['Datetime', 'Text', 'Username', 'Likes'])
        
        # Log the tweet dataframe
        #logging.info(f'Tweet DataFrame:\n{tweet_df}')

        from pandasai import PandasAI
        from pandasai.llm.openai import OpenAI

        # Instantiate a LLM
        load_dotenv()
        openai_key = os.getenv('OPENAI_API_KEY')
        llm = OpenAI()
        pandas_ai = PandasAI(llm, conversational=True, enable_cache=False)

        # Log the prompt and run pandas_ai
        prompt = 'You are a talk show host and youre about to interview a very famous startup founder. Based on the tweets of this person, generate five potential interesting questions that a wide range of people might find interesting.'
        #logging.info(f'Prompt: {prompt}')
        questions = pandas_ai(tweet_df, prompt=prompt)
        questionlist = re.findall(r'\d+\.\s+(.*)', questions)

        return questionlist
    
    except Exception:
        print('Twitter error')
        return []

test_handle = '@susantoscott' 
#print(twitter(test_handle))