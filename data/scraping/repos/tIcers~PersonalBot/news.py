import os 
import discord
from openai.api_resources import engine 
import requests 
import openai
from discord.ext import commands 
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from pprint import pprint

load_dotenv()


OPEN_API = os.environ.get("OPEN_API")

# bot = commands.Bot(command_prefix='!')


intents = discord.Intents.default()
intents.members = True

client = discord.Client(intents=intents)
NEWS_URL='https://www.dailyfx.com/usd-jpy/news-and-analysis'
response = requests.get(NEWS_URL)
print(response.text) # this is for testing response


# def scrape_news():
#     NEWS_URL='https://www.dailyfx.com/usd-jpy/news-and-analysis'
#     response = requests.get(NEWS_URL)
#     print(response.text) # this is for testing response
#
#     soup = BeautifulSoup(response.content, 'html.parser')
#
#     news_articles = []
#     open_api_key = OPEN_API
#
#     summaries = []
#
#     for article in news_articles:
#         summary = openai.Completion.create(
#                 engine = 'davinci-002',
#                 prompt=f"Summarize the following news article:\n{article}",
#                 max_token = 100,
#                 )
#
#
#         summaries.append(summary.choices[0].text)
#         print(summaries)
#
#     return summaries



# def summrize_news(news):
#     openai.api_key = OPEN_API
#
#     model_engine = 'text-davinci-002'
