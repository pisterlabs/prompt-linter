import cohere
import pandas as pd
import os
from nltk import tokenize
from typing import List
from tqdm import tqdm
import numpy as np

# nltk.download('punkt')

# initialize the Cohere Client with an API Key
API_KEY = os.getenv('COHERE_API_KEY')
co = cohere.Client(API_KEY)

def extract_embedding(sentences: List[str]):
    response = co.embed(model='small', texts=sentences)
    
    return response.embeddings

# Extract sentence embeddings and store them in .npy files
# Store sentences in csv file
records = pd.read_csv("bgg_2000.csv")
sentence_df = None
progress_bar = tqdm(records.iterrows())
for idx, record in progress_bar:
    progress_bar.set_description("Extracting embedding for game %s" % idx)
    description = record['description']
    sentences = tokenize.sent_tokenize(description)
    df = pd.DataFrame({'uid': [idx] * len(sentences), "sentence": sentences})
    if sentence_df is None:
        sentence_df = df
    else:
        sentence_df = pd.concat([sentence_df, df])

    embeddings = extract_embedding(sentences)
    np.save('data/%s.npy' % idx, embeddings)

# Save all sentences to file
sentence_df.to_csv("bgg_2000_sentences.csv", index=False)

# Save all embeddings into a single file
sentence_embeddings = None
for idx in range(records.shape[0]):
    embedding = np.load('data/%s.npy' % idx)
    if sentence_embeddings is None:
        sentence_embeddings = embedding
    else:
        sentence_embeddings = np.concatenate([sentence_embeddings, embedding])

np.save('data/all_embed.npy', sentence_embeddings)
