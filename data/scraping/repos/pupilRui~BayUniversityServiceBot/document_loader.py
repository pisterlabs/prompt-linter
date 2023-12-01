#!/usr/bin/env python
# coding: utf-8

import os
import openai
import sys
import requests
sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = os.environ['API_KEY']
goog_api_key    = os.environ['GOOG_API_KEY']


from langchain.document_loaders import PyPDFLoader
loader = PyPDFLoader("data/pdf/2023Catalog.pdf")
pages = loader.load()
print("loaded pdf, metadata:\n")
print(pages[0].metadata)


# ## YouTube

# In[ ]:


from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import OpenAIWhisperParser
from langchain.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader


# In[ ]:


# ! pip install yt_dlp
# ! pip install pydub


# **Note**: This can take several minutes to complete.

# In[ ]:


def get_channel_videos(channel_id, api_key):
    url = f'https://www.googleapis.com/youtube/v3/search?key={api_key}&channelId={channel_id}&part=snippet,id&order=date&maxResults=50'
    response = requests.get(url)
    data = response.json()
    videos = []

    for item in data['items']:
        if item['id']['kind'] == 'youtube#video':
            videos.append('https://www.youtube.com/watch?v=' + item['id']['videoId'])

    return videos


def search_videos(query, api_key):
    url = f'https://www.googleapis.com/youtube/v3/search?key={api_key}&q={query}&part=snippet,id&order=relevance&maxResults=50'
    response = requests.get(url)
    data = response.json()
    videos = []

    if 'items' in data:
        for item in data['items']:
            if item['id']['kind'] == 'youtube#video':
                videos.append('https://www.youtube.com/watch?v=' + item['id']['videoId'])

    return videos

channel_id = 'UCq476UNYNtbp-flsx-B6kLw'
videos1 = get_channel_videos(channel_id, goog_api_key)
search_query1 = 'SFBU DeepPiCar'
videos2 = search_videos(search_query1, goog_api_key)
search_query2 = 'SFBU'
videos3 = search_videos(search_query2, goog_api_key)

videos_special = ['https://www.youtube.com/watch?v=kuZNIvdwnMc', 'https://www.youtube.com/watch?v=1gJcCM5G32k', 'https://www.youtube.com/watch?v=hZE5fT7CVdo']

# Merge all the lists
all_videos = videos1 + videos2 + videos3 + videos_special

# Remove duplicates by converting the list to a set and back to a list
unique_videos = list(set(all_videos))


for url in unique_videos:
    save_dir="data/youtube/"
    print(f"Loading video: {url}")
    loader = GenericLoader(
        YoutubeAudioLoader([url],save_dir),
        OpenAIWhisperParser()
    )
    docs = loader.load()
    print("loaded metadata:\n")
    print(docs[0].metadata)


# ## URLs

# In[ ]:


from langchain.document_loaders import WebBaseLoader

urls = ['https://www.sfbu.edu/admissions/student-health-insurance', 'https://www.sfbu.edu/about-us', 'https://www.sfbu.edu/admissions', 'https://www.sfbu.edu/academics', 'https://www.sfbu.edu/student-life', 'https://www.sfbu.edu/contact-us']

for url in urls:
    print(f"Loading url: {url}")
    loader = WebBaseLoader(url)
    docs = loader.load()
    print("loaded metadata:\n")
    print(docs[0].metadata)