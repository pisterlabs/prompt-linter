# from .Whatsapp import *
import os
import openai
import json
import feedparser
import requests

from bs4 import BeautifulSoup

from dotenv import load_dotenv

class NewsReader:

    pre_prompt = ""

    def prepare(self):
        # Biased example
        self.pre_prompt = "Please respond bullish, bearish, neutral"

    def get_feed_from_FX_street(self):

        entries = []

        url = f"https://rss.app/feeds/tGOlxdesgAN5b5ay.xml"

        count = 5

        # Symbol to filter by
        # symbol = 'GBP/USD'

        # Parse the feed
        feed = feedparser.parse(url)

        # Loop through the articles and print the titles, summaries, and links
        for i, entry in enumerate(feed.entries):
            if i > count:
                break
            # if symbol in entry.summary:
            # print(entry.title)
            # print(entry.summary)
            # print(entry.link)
            # print('-' * 80)
            # print(self.get_response("GBP/USD", f"{entry.title} - {entry.summary}"))
            # print('-' * 80)

            # URL of the FXStreet news article
            # url = 'https://www.fxstreet.com/news/pound-sterling-price-news-and-forecast-gbp-usd-remains-indecisive-as-the-boe-monetary-policy-decision-looms-202305110344'

            # # Make a GET request to the page
            # response = requests.get(url)

            # # Parse the HTML content of the page with BeautifulSoup
            # soup = BeautifulSoup(response.content, 'html.parser')

            # # Extract the title and publication date of the article
            # title = soup.find('h1', class_='fxs_headline_large').text.strip()
            # date = soup.find('time')['datetime']

            # # Extract the main content of the article
            # content = ''
            # paragraphs = soup.find_all('div', class_='fxs_article_body')
            # for p in paragraphs:
            #     content += p.text.strip() + '\n\n'

            # # Print out the title, date, and content of the article
            # print(title)
            # print(date)
            # print(content)

            entries.append({"symbol": "GBP/USD", "news": f"{entry.title} - {entry.summary}", "result": self.get_response("GBP/USD", f"{entry.title} - {entry.summary}")})

        return entries
                
    def get_feed(self):
        entries = []
        
        ticker = 'GBPUSD=X'

        rssfeedurl = 'https://feeds.finance.yahoo.com/rss/2.0/headline?s=%s&region=US&lang=en-US'%ticker

        NewsFeed = feedparser.parse(rssfeedurl)
        NewsFeed.keys()
        
        range_max = len(NewsFeed.entries)

        if range_max > 3:
            range_max = 3

        for i in range(range_max):
            message = f'{NewsFeed.entries[i].title} {NewsFeed.entries[i].summary}'
            entries.append({"symbol": "GBPUSD", "news": message, "result": self.get_response("GBPUSD", message)})

        ticker = 'EURUSD=X'

        rssfeedurl = 'https://feeds.finance.yahoo.com/rss/2.0/headline?s=%s&region=US&lang=en-US'%ticker

        NewsFeed = feedparser.parse(rssfeedurl)
        NewsFeed.keys()
        
        range_max = len(NewsFeed.entries)

        if range_max > 3:
            range_max = 3

        for i in range(range_max):
            message = f'{NewsFeed.entries[i].title} {NewsFeed.entries[i].summary}'
            entries.append({"symbol": "EURUSD", "news": message, "result": self.get_response("EURUSD", message)})

        ticker = 'GC=F'

        rssfeedurl = 'https://feeds.finance.yahoo.com/rss/2.0/headline?s=%s&region=US&lang=en-US'%ticker

        NewsFeed = feedparser.parse(rssfeedurl)
        NewsFeed.keys()
        
        range_max = len(NewsFeed.entries)

        if range_max > 3:
            range_max = 3

        for i in range(range_max):
            message = f'{NewsFeed.entries[i].title} {NewsFeed.entries[i].summary}'
            entries.append({"symbol": "GOLD", "news": message, "result": self.get_response("GOLD", message)})

        return entries

    def get_response(self, ticker, input):

        input = input.replace("\"", "\n")

        # response = openai.Completion.create(
        #     model="text-davinci-003",
        #     prompt=f"{self.pre_prompt}\n {input}\n",
        #     max_tokens=800,
        #     temperature=0
        # )

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                    {"role": "user", "content": f"Given the ticker {ticker}, please respond only bullish, bearish or neutral to the following news: {input}\n"}
                ]
        )

        # response = openai.ChatCompletion.create(
        #     model="gpt-3.5-turbo",
        #     messages=[
        #         {"role": "user", "content": f"{self.pre_prompt}\n {input}\n"}
        #     ]
        # )

        if len(response.choices) > 0:
            return response.choices[0].message.content.replace("\n","")
        else:
            return "Error: No response could be obtained by Taylor this time."

    def __init__(self, *args, **kwargs):
        load_dotenv()

        openai.api_key = os.getenv("OPENAI_API_KEY")

        self.prepare()

        super().__init__(*args, **kwargs)
