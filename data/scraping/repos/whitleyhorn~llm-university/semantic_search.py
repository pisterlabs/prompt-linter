import cohere
from utils.cohere_utils import co
import numpy as np
import re
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
import umap
import altair as alt
from sklearn.metrics.pairwise import cosine_similarity
from annoy import AnnoyIndex
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_colwidth', None)

# Load the dataset
dataset = load_dataset("trec", split="train")
# Import into a pandas dataframe, take only the first 1000 rows
df = pd.DataFrame(dataset)[:1000]

# get the embeddings
embeds = co.embed(texts=list(df["text"]),
                  model='embed-english-v2.0').embeddings

# Create the search index, pass the size of the embeddings
search_index = AnnoyIndex(np.array(embeds).shape[1], 'angular')

# Add the embeddings to the search index
for i in range(len(embeds)):
    search_index.add_item(i, embeds[i])
search_index.build(10) # 10 trees
search_index.save('test.ann')

query = input("Enter your query: ")

# Get the query's embedding
query_embed = co.embed(texts=[query],
                  model="embed-english-v2.0").embeddings

# Retrieve the nearest neighbors
similar_item_ids = search_index.get_nns_by_vector(query_embed[0],10,
                                                include_distances=True)
# Format the results
results = pd.DataFrame(data={'texts': df.iloc[similar_item_ids[0]]['text'], 
                             'distance': similar_item_ids[1]})


print(f"Query:'{query}'\nNearest neighbors:")
print(results)
