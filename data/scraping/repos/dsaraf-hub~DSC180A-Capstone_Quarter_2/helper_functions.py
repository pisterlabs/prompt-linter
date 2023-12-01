from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
import pandas as pd
import openai
import os
import re

# function to check if any word from the list is present in the string
def is_word_present(string, words_list):
    return any(word in string for word in words_list)

# function to get back embeddings from OpenAI using the text-embedding-ada-002 engine
def get_embedding(text: str, engine="text-embedding-ada-002"):
  # replace newlines, which can negatively affect performance.
  text = [q.replace("\n", " ") for q in text]
  return [j['embedding'] for j in openai.Embedding.create(input=text, engine=engine)["data"]]

# function to chunk out a list in groups of 10
def chunks(input_list):
    output_lists = []
    for i in range(0, len(input_list), 30):
        output_lists.append(input_list[i:i+30])
    return output_lists

# function to extract tickers from a series of tweets
def extract_tickers(tweets, words_to_search):
    tickers = []
    for tweet in tweets:
      found = []
      for word in words_to_search:
        if word in tweet:
          found.append(word)
      tickers.append(found)
    return pd.Series(tickers)

def classify_tweet(ticker, text):
  try:
    response = openai.Completion.create(
      model="text-davinci-003",
      prompt=f"Is the following tweet positive, negative, or neutral news for {ticker}: \"{text}",
      temperature=0.7,
      max_tokens=256,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0
    )
    resp = response['choices'][0]['text'].strip().lower()
  except:
    response = openai.Completion.create(
      model="text-davinci-003",
      prompt=f"Is the following tweet positive, negative, or neutral news for {ticker}: \"{text}",
      temperature=0.7,
      max_tokens=256,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0
    )
    resp = response['choices'][0]['text'].strip().lower()
  if 'positive' in resp:
    return 'positive'
  elif 'negative' in resp:
    return 'negative'
  else:
    return 'neutral'