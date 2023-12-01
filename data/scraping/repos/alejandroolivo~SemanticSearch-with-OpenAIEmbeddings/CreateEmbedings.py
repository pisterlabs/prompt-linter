import openai
import pandas as pd
import numpy as np
import configparser
from getpass import getpass
from openai.embeddings_utils import get_embedding

# Read config
config = configparser.ConfigParser()
config.read('config.ini')

# Write KEY for openAI API
openai.api_key = config['openai']['api_key']

# Read the data from the csv file
df = pd.read_csv('.\Data\Tags.csv')
print(df)

#Calculate word embeddings for each tag
df['embedding'] = df['tag'].apply(lambda x: get_embedding(x, engine='text-embedding-ada-002'))
df.to_csv('.\Data\word_embeddings.csv')