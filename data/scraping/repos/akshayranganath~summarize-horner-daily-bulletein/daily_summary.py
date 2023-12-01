#!/usr/bin/env python
# coding: utf-8

# In[75]:


import requests
from bs4 import BeautifulSoup
import openai
from datetime import date
import os


# In[76]:


openai.api_key = os.environ.get('OPENAPI_KEY') or print('No OPEN API Key found!')
bot_token = os.environ.get('BOT_TOKEN') or print('No Telegram bot token found!')
chat_id = os.environ.get('CHAT_ID') or print('No Telegram chat_id found!')

# In[77]:


def get_today():
    today = date.today()
    return f'{today.month:02}-{today.day:02}-{today.year}'


# In[78]:


def get_completion(prompt=None, model="gpt-3.5-turbo",test=True):
    messages = [
        {
            "role": "system",
            "content": "You are a text summarizing service. Given the daily bulletein, your job is to condense the message so that it is easy to read."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]
    response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0, # this is the degree of randomness of the model's output
    )
    #print(response.choices[0].message["content"])
    return response.choices[0].message["content"]


# In[79]:


# Web scraping
headers = {
    "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36",
    "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    "accept-language": "en-US,en;q=0.9",
    "cache-control": "max-age=0",
    "sec-ch-ua": "\"Chromium\";v=\"116\", \"Not)A;Brand\";v=\"24\", \"Google Chrome\";v=\"116\"",
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": "\"macOS\"",
    "sec-fetch-dest": "document",
    "sec-fetch-mode": "navigate",
    "sec-fetch-site": "none",
    "sec-fetch-user": "?1",
    "sec-gpc": "1",
    "upgrade-insecure-requests": "1"
  }
url = f'https://fremontunified.org/horner/news/daily-bulletin-{get_today()}/'
response = requests.get(url,headers=headers)
soup = BeautifulSoup(response.text, 'html.parser')
text = soup.get_text().replace('\n','')
if text.startswith('Page Not Found'):
    html_content = f'No updates for {get_today()}!'
else:
    html_content = get_completion(text)

print(html_content)
# In[80]:


# Send scraped webpage content to Telegram channel
telegram_api_url = f'https://api.telegram.org/bot{bot_token}/sendMessage'
message_text = f"Scraped webpage content:\n\n{html_content}\n\nFull notification: {url}"

params = {
    'chat_id': chat_id,
    'text': message_text,
    'parse_mode': 'HTML'
}

response = requests.post(telegram_api_url, params=params)
if response.status_code == 200:
    print("Message sent successfully to Telegram channel!")
else:
    print(f"{response.status_code}: Failed to send message to Telegram channel.")
    print(response.json())

