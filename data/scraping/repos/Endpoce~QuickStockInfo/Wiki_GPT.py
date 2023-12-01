import openai
import requests
import wikipedia
from Gen_Files.GetArticles import *
import os
from dotenv import load_dotenv
import yfinance as yf


def get_wiki_info(query):
    results = wikipedia.search(query)
    if results != None:
        first_result = results[0]  # get the first result
        try:
            # get the page of the first result
            page = wikipedia.page(first_result)
            url = page.url  # get the url of the page
            return url  # return the content of the page
        except wikipedia.DisambiguationError as e:
            print(
                f"Disambiguation page found, consider choosing a specific title from: {e.options}")
        except wikipedia.PageError:
            print("Page not found on Wikipedia")
    else:
        return None  # return None if no results found

def summarize_article(url):
    # Initialize a new chat model
    chat_model = openai.Completion.create(
        model="text-davinci-003",
        prompt=f"Write: '--Summary--'. Write 'Sentiment: ' and give the sentiment of te article. Two lines down, Write a summary of the provided article. Then write on a new line: '--Additional Info--'. Then return a list of the main points in the provided article, one on each line. Limit each list item to 100 words, and return no more than 10 points per list. URL: {url}\n\nSummary:",
        temperature=0.3,
        max_tokens=300
    )

    return chat_model['choices'][0]['text']
