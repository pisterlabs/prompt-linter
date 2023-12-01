import os
import pandas as pd
import numpy as np
import openai
from openai.embeddings_utils import get_embedding
import configparser

# Read config
config = configparser.ConfigParser()
config.read('config.ini')

# Write KEY for openAI API
os.environ['OPENAI_API_KEY'] = config['openai']['api_key']

# Define the root directory path for word embedding batches
rootdir = '.\Data\Word Embeddings Batches'

# Input a search term
search_term = input('Enter a search term: ')
search_term_vector = get_embedding(search_term, engine="text-embedding-ada-002")

# Create an empty list to store the search results
search_results = []

# Iterate over each file in the root directory
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        # Check if the file is a word embedding batch file
        if file.startswith('word_embeddings_batch'):
            # Read the data from the file
            filepath = os.path.join(subdir, file)
            df = pd.read_csv(filepath)
            df['embedding'] = df['embedding'].apply(eval).apply(np.array)

            # Calculate the cosine similarity between the search term and each tag
            df['similarity'] = df['embedding'].apply(lambda x: np.dot(x, search_term_vector) / (np.linalg.norm(x) * np.linalg.norm(search_term_vector)))

            # Add the search results to the list
            search_results.append(df)

# Combine the search results into a single dataframe
df = pd.concat(search_results)

# Sort the tags by similarity
df = df.sort_values(by='similarity', ascending=False)

# Print the top 10 tags
print(df.head(10))

# Save the results to a CSV file
df.to_csv('.\Data\search_results.csv')
