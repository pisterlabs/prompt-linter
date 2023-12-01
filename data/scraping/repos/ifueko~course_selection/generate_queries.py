import numpy as np
import os
import openai
import pandas as pd
import time
import traceback
from openai.embeddings_utils import get_embedding
from tqdm import tqdm

openai.api_key = os.getenv("OPENAI_API_KEY")
CHUNK_SIZE=2000

df = pd.read_csv("queries.csv")
df_keys = ["text","embedding"]
df['embedding_combined'] = [[] for _ in range(len(df))]
pbar = tqdm(total=len(df))
model = openai.Embedding.create(model="text-embedding-ada-002", input=list(df['text']))
for i in range(len(df)):
    df.loc[i]['embedding_combined'] = model.data[i]['embedding']

print(df)
df.to_csv("queries_with_embeddings.csv", index=False)
