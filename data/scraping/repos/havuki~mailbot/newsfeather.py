import openai
import requests
import os
from googleapiclient.discovery import build
from datetime import datetime, timedelta

def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

def save_file(filepath, content):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        outfile.write(content)

GOOGLE_SEARCH_API_KEY = open_file('googleapikey.txt')
GOOGLE_SEARCH_CX = open_file('googlecx.txt')

openai.api_key = open_file('openaiapikey.txt')

conversation = []

def fetch_ai_news():
    # Build the Google Search service
    search_service = build("customsearch", "v1", developerKey=GOOGLE_SEARCH_API_KEY)

    # Define the search query
    query = "AI"

    # Execute the search
    results = search_service.cse().list(q=query, cx=GOOGLE_SEARCH_CX, num=6).execute()

    # Extract relevant information from the search results
    news_items = [{'title': result['title'], 'snippet': result['snippet'], 'url': result['link']} for result in results['items']]

    return news_items

# Fetch AI news
news_items = fetch_ai_news()

# Extract news headlines
headlines = [item['title'] for item in news_items]

