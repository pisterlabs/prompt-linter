import openai
from openai.embeddings_utils import get_embedding
openai.api_key = "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"
import tqdm as tqdm
# Ensure you have your API key set in your environment per the README: https://github.com/openai/openai-python#usage

import pandas as pd

input_datapath = 'complete-courses.csv'  # to save space, we provide a pre-filtered dataset
df = pd.read_csv(input_datapath)

df = df.dropna()
df['combined'] = "Number: " + df['number'].str.strip() + "; Title: " + df['name'].str.strip() + "; Content: " + df['description'].str.strip()
df.head(2)

print("Preparing embeddings ...")
# This will take just between 5 and 10 minutes
#df['ada_similarity'] = df.combined.apply(lambda x: get_embedding(x, engine='text-embedding-ada-002'))

from tqdm import tqdm, tqdm_pandas
tqdm_pandas(tqdm())

df['ada_search'] = df.combined.progress_apply(lambda x: get_embedding(x, engine='text-embedding-ada-002'))
df.to_csv('complete-courses-embed.csv')