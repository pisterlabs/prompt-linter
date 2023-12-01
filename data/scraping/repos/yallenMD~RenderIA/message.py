#Import required modules
from linebot import (LineBotApi, WebhookHandler)
from linebot.exceptions import (InvalidSignatureError)
from linebot.models import *
import openai
import tempfile, os
import datetime
import time
import requests
import json
from newsapi import NewsApiClient

#Initiate enviornment variables as API keys
#Channel Access Token
line_bot_api = LineBotApi(os.getenv('CHANNEL_ACCESS_TOKEN'))
# Channel Secret
handler = WebhookHandler(os.getenv('CHANNEL_SECRET'))
# Twelve API Key
api_key = os.getenv('TWELVEDATA_API_KEY')
# Open AI API Key
openai.api_key = os.getenv('OPENAI_API_KEY')
#News API Key
news_key = os.getenv('NEWS_API_KEY')
#Init News API Key
newsapi = NewsApiClient(api_key=news_key)

def GPT_message(text):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", 
        messages=[{"role": "system", "content": 'You are a helpful financial analyst who understands stocks and crypto. Pretend like you are texting someone and limit the text messages to an appropriate length.'},
                    {"role": "user", "content": text}
                 ])

    # 重組回應
    answer = response['choices'][0]['message']['content']
    return answer

def summarize(text):
    if len(text) <= 40:
        return text
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", 
        messages=[{"role": "system", "content": 'You are a professional text summarizer that will summarize the news article titles and descriptions you are given into strictly 40 characters or less. (White spaces are considered a character)'},
                    {"role": "user", "content": text}
                 ])
    # 重組回應
    answer = response['choices'][0]['message']['content']
    return answer if len(answer) <= 40 else answer[:37]+'...'

def price(ticker,api_key):
    url = f"https://api.twelvedata.com/price?symbol={ticker}&apikey={api_key}"
    response = requests.get(url)
    response = response.json()
    price = response['price'][:-3]
    return f"The price of {ticker} is {price} USD"

def exchange_rate(exchange_from,exchange_to,api_key):
    url = f"https://api.twelvedata.com/exchange_rate?symbol={exchange_from}/{exchange_to}&apikey={api_key}"
    response = requests.get(url)
    response = response.json()
    exchange_rate = response['rate']
    return exchange_rate

def currency_conversion(exchange_from,exchange_to,amount,api_key):
    url = f"https://api.twelvedata.com/currency_conversion?symbol={exchange_from}/{exchange_to}&amount={amount}&apikey={api_key}"
    response = requests.get(url)
    response = response.json()
    original_amount = amount
    new_amount = response['amount']
    return f"{original_amount} {exchange_from} is equivalent to {new_amount} {exchange_to}"

def news(subject,news_key):
    url = f"https://newsapi.org/v2/everything?q={subject}&apiKey={news_key}"
    response = requests.get(url)
    response = response.json()
    first_five_articles = [(article['title'], article['url'], article['urlToImage']) for article in response['articles'][:5]]
    articles = ""
    for title, url, urlToImage in first_five_articles:
        articles += f"Title: {title}\nURL: {url}\nImage: {urlToImage}\n---\n"
    return articles

def news_carousel(subject,news_key):
    url = f"https://newsapi.org/v2/everything?q={subject}&apiKey={news_key}"
    response = requests.get(url)
    response = response.json()

    articles = response['articles'][:5]

    titles = [summarize(article['title']) for article in articles]
    descriptions = [summarize(article['description']) for article in articles]

    urls = [article['url'] for article in articles]
    images = [article['urlToImage'] for article in articles]


    message = TemplateSendMessage(
        alt_text='Top 5 headlines requested by you',
        template=CarouselTemplate(
            columns=[CarouselColumn(thumbnail_image_url=images[i],title=titles[i],text=descriptions[i],actions=[URITemplateAction(label='Link to article',uri=urls[i])]) for i in range(5)]
        )
    )
    return message

def Confirm():
    message = TemplateSendMessage(
        alt_text='Would you like to know what I can do?',
        template=ConfirmTemplate(
            text="Would you like to know what I can do?",
            actions=[
                MessageTemplateAction(
                    label="Yes!",
                    text="I would like to know what you can do!"
                ),
                    MessageTemplateAction(
                    label="No thanks.",
                    text="It's all good"
                )
            ]
        )
    )
    return message