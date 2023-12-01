import cohere
import numpy as np
import re
import os
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
result = os.path.dirname(os.path.realpath('__file__'))


api_key = '22L1f26PPdIZVcoLeERyldwGIRVGXcUvqfaQNItT'
co = cohere.Client(api_key)
# Get dataset
df = pd.read_csv("../data/internships.csv")
embeds = co.embed(texts=list(df['job_description']),
                  model='embed-english-v2.0').embeddings
all_embeddings = np.array(embeds)

relative_path = "../data/embeddings.npy"
full_path = os.path.join(result, relative_path)
np.save(full_path, all_embeddings)


search_index = AnnoyIndex(np.array(embeds).shape[1], 'angular')
# Add all the vectors to the search index
for i in range(len(embeds)):
    search_index.add_item(i, embeds[i])
search_index.build(3)


query = """Advocate for usability and consistency across the Firm’s products
Build Single Page Applications using SharePoint 2010/2013’s Client Object Model, SPServices, and jQuery
Create cutting edge mobile, web, and desktop interface designs that meet today’s evolving industry standards, business requirements, and are consistent with the KP brand
Oversee others’ designs and be able to critique and guide them
Collaborate and partner with taxonomists, engineers, and program management to develop an overarching UX strategy and implementation plan for Taxonomy Tools and Automation
Help visualize, communicate, and implement designs and assets
Bring strong, user-centered design skills and business knowledge to translate complex workflows into exceptional design solutions at scale"""
# Get the query's embedding
query_embed = co.embed(texts=[query], model="embed-english-v2.0").embeddings

# Retrieve the nearest neighbors
similar_item_ids = search_index.get_nns_by_vector(
    query_embed[0], 10, include_distances=True)
# Format the results
results = pd.DataFrame(
    data={'texts': df.iloc[similar_item_ids[0]]['job_description'], 'distance': similar_item_ids[1]})

print(f"Query:'{query}'\nNearest neighbors:")
results
