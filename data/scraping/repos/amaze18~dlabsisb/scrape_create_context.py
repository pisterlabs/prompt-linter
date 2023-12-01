 # -*- coding: utf-8 -*-
"""ISB chatbot.ipynb



Original file is located at
    https://colab.research.google.com/drive/1GYmsZSR4MWuvORNpSWFWrXz79lQKb6oc
"""

"""# Scrape"""

# Regex to match a URL
# HTTP_URL_PATTERN = r'^http[s]{0,1}://.+$'

# Define root domain to crawl
domain = "i-venture.org"
sitemap_url = "https://i-venture.org/sitemap.xml"
full_url = "https://i-venture.org/"

import os

RESULTS_DIR = "scraped_files/"
os.makedirs(RESULTS_DIR, exist_ok=True)

import requests
import re
import urllib.request
from bs4 import BeautifulSoup
from collections import deque
from html.parser import HTMLParser
from urllib.parse import urlparse
import os
import pandas as pd
import numpy as np

def get_sitemap(url=sitemap_url):
    try:
        with urllib.request.urlopen(url) as response:
            xml = BeautifulSoup(response,
                                'lxml-xml',
                                from_encoding=response.info().get_param('charset'))

        urls = xml.find_all("url")
        locs = []

        for url in urls:

            if xml.find("loc"):
                loc = url.findNext("loc").text
                locs.append(loc)

        return locs
    except Exception as e:
        print(e)
        return []


def crawl(url):
    # Parse the URL and get the domain
    # local_domain = urlparse(url).netloc

    queue = deque(get_sitemap())

    os.makedirs(RESULTS_DIR + "text/", exist_ok=True)
    os.makedirs(RESULTS_DIR + "processed", exist_ok=True)

    # While the queue is not empty, continue crawling
    while queue:
        # Get the next URL from the queue
        url = queue.pop()
        print(url) # for debugging and to see the progress

        # Save text from the url to a <url>.txt file
        with open(f'{RESULTS_DIR}text/'+ url.strip("/").replace("/", "_") + ".txt", "w", encoding="UTF-8") as f:

            soup = BeautifulSoup(requests.get(url).text, "html.parser")
            text = soup.get_text()

            # If the crawler gets to a page that requires JavaScript, it will stop the crawl
            if ("You need to enable JavaScript to run this app." in text):
                print("Unable to parse page " + url + " due to JavaScript being required")

            f.write(text)

        # # Get the hyperlinks from the URL and add them to the queue
        # for link in get_domain_hyperlinks(local_domain, url):
        #     if link not in seen:
        #         queue.append(link)
        #         seen.add(link)

def remove_newlines(serie):
    serie = serie.str.replace('\n', ' ')
    serie = serie.str.replace('\\n', ' ')
    serie = serie.str.replace('  ', ' ')
    serie = serie.str.replace('  ', ' ')
    return serie


def get_df():
    # Create a list to store the text files
    texts=[]

    for file in os.listdir(RESULTS_DIR + "text/"):
        with open(RESULTS_DIR + "text/" + "/" + file, "r", encoding="UTF-8") as f:
            text = f.read()

            # Omit the first 11 lines and the last 4 lines, then replace -, _, and #update with spaces.
            texts.append((file.replace('#update',''), text))

    # Create a dataframe from the list of texts
    df = pd.DataFrame(texts, columns = ['fname', 'text'])

    # Set the text column to be the raw text with the newlines removed
    df['text'] = df.fname + ". " + remove_newlines(df.text)
    return df



"""# Create Embeddings

## Clean
"""


import tiktoken
from openai.embeddings_utils import distances_from_embeddings, cosine_similarity

# Load the cl100k_base tokenizer which is designed to work with the ada-002 model
tokenizer = tiktoken.get_encoding("cl100k_base")

df = pd.read_csv(RESULTS_DIR + 'processed/scraped.csv', index_col=0)
df.columns = ['title', 'text']

# Tokenize the text and save the number of tokens to a new column
df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))

# Visualize the distribution of the number of tokens per row using a histogram
df.n_tokens.hist()

max_tokens = 500

# Function to split the text into chunks of a maximum number of tokens
def split_into_many(text, max_tokens = max_tokens):

    # Split the text into sentences
    sentences = text.split('. ')

    # Get the number of tokens for each sentence
    n_tokens = [len(tokenizer.encode(" " + sentence)) for sentence in sentences]

    chunks = []
    tokens_so_far = 0
    chunk = []

    # Loop through the sentences and tokens joined together in a tuple
    for sentence, token in zip(sentences, n_tokens):

        # If the number of tokens so far plus the number of tokens in the current sentence is greater
        # than the max number of tokens, then add the chunk to the list of chunks and reset
        # the chunk and tokens so far
        if tokens_so_far + token > max_tokens:
            chunks.append(". ".join(chunk) + ".")
            chunk = []
            tokens_so_far = 0

        # If the number of tokens in the current sentence is greater than the max number of
        # tokens, go to the next sentence
        if token > max_tokens:
            continue

        # Otherwise, add the sentence to the chunk and add the number of tokens to the total
        chunk.append(sentence)
        tokens_so_far += token + 1

    # Add the last chunk to the list of chunks
    if chunk:
        chunks.append(". ".join(chunk) + ".")

    return chunks

def shorten(df):
    shortened = []

    # Loop through the dataframe
    for row in df.iterrows():

        # If the text is None, go to the next row
        if row[1]['text'] is None:
            continue

        # If the number of tokens is greater than the max number of tokens, split the text into chunks
        if row[1]['n_tokens'] > max_tokens:
            shortened += split_into_many(row[1]['text'])

        # Otherwise, add the text to the list of shortened texts
        else:
            shortened.append( row[1]['text'] )

    new_df = pd.DataFrame(shortened, columns = ['text'])
    new_df['n_tokens'] = new_df.text.apply(lambda x: len(tokenizer.encode(x)))
    return new_df

df = shorten(df)
df.n_tokens.hist()

"""## Create embeds"""



import openai
from dotenv import load_dotenv
load_dotenv()

SECRET_IN_ENV = False

import os
SECRET_TOKEN = os.getenv("SECRET_TOKEN_GPT3.5")

openai.api_key = SECRET_TOKEN

# Note that you may run into rate limit issues depending on how many files you try to embed
# Please check rate limit guide to learn more on how to handle this: https://platform.openai.com/docs/guides/rate-limits

df['embeddings'] = df.text.apply(lambda x: openai.Embedding.create(input=x, engine='text-embedding-ada-002')['data'][0]['embedding'])
df.to_csv('processed/embeddings.csv')
df.head()

"""# QnA"""

from ast import literal_eval

df = pd.read_csv('processed/embeddings.csv', index_col=0)
df['embeddings'] = df['embeddings'].apply(literal_eval).apply(np.array)


def create_context(
    question, df, max_len=1800, size="ada"
):
    """
    Create a context for a question by finding the most similar context from the dataframe
    """

    # Get the embeddings for the question
    q_embeddings = openai.Embedding.create(input=question, engine='text-embedding-ada-002')['data'][0]['embedding']

    # Get the distances from the embeddings
    df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].values, distance_metric='cosine')


    returns = []
    cur_len = 0

    # Sort by distance and add the text to the context until the context is too long
    for i, row in df.sort_values('distances', ascending=True).iterrows():

        # Add the length of the text to the current length
        cur_len += row['n_tokens'] + 4

        # If the context is too long, break
        if cur_len > max_len:
            break

        # Else add it to the text that is being returned
        returns.append(row["text"])

    # Return the context
    return "\n\n###\n\n".join(returns)

def answer_question(
    df,
    model="text-davinci-003",
    question="Am I allowed to publish model outputs to Twitter, without a human review?",
    max_len=1800,
    size="ada",
    debug=False,
    max_tokens=150,
    stop_sequence=None
):
    """
    Answer a question based on the most similar context from the dataframe texts
    """
    context = create_context(
        question,
        df,
        max_len=max_len,
        size=size,
    )
    # If debug, print the raw model response
    if debug:
        print("Context:\n" + context)
        print("\n\n")

    try:
        # Create a completions using the questin and context
        response = openai.Completion.create(
            prompt=f"Answer the question based on the context below, and if the question can't be answered based on the context, say \"I don't know\"\n\nContext: {context}\n\n---\n\nQuestion: {question}\nAnswer:",
            temperature=0,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
            model=model,
        )
        return response["choices"][0]["text"].strip()
    except Exception as e:
        print(e)
        return ""

print(answer_question(df, question="What day is it?", debug=False))

print(answer_question(df, question="What is our newest embeddings model?"))

