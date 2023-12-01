import requests
import re
import urllib.request
from bs4 import BeautifulSoup
from collections import deque
from html.parser import HTMLParser
from urllib.parse import urlparse
import os
import pandas as pd
import tiktoken
import openai
from openai.embeddings_utils import distances_from_embeddings
import numpy as np
from openai.embeddings_utils import distances_from_embeddings, cosine_similarity

def remove_newlines(serie):
    serie['colname'] = serie['colname'].str.replace('\n', ' ')
    serie['colname'] = serie['colname'].str.replace('\\n', ' ')
    serie['colname'] = serie['colname'].str.replace('  ', ' ')
    serie['colname'] = serie['colname'].str.replace('  ', ' ')
    return serie

# Create a list to store the text files
texts=[]

# Get all the text files in the text directory
for file in os.listdir("text/" + domain + "/"):

    # Open the file and read the text
    with open("text/" + domain + "/" + file, "r", encoding="UTF-8") as f:
        text = f.read()

        # Omit the first 11 lines and the last 4 lines, then replace -, _, and #update with spaces.
        texts.append((file[11:-4].replace('-',' ').replace('_', ' ').replace('#update',''), text))

# Create a dataframe from the list of texts
df = pd.DataFrame(texts, columns = ['fname', 'text'])

# Set the text column to be the raw text with the newlines removed
df['text'] = df.fname # + ". " + remove_newlines(df.text)
df.to_csv('processed/scraped.csv')
df.head()