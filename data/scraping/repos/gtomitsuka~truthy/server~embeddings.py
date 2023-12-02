import cohere
from annoy import AnnoyIndex
import os
import json

import numpy as np
import pandas as pd

cohere_key = os.getenv('COHERE_KEY')
co = cohere.Client(cohere_key)

with open('embeddings.json') as embeddings_file:
  sources_json = embeddings_file.read()

sources = json.loads(sources_json)['information']
source_df = pd.DataFrame.from_dict(sources)

embeds = co.embed(texts=list(source_df['claim']),
                  model='embed-english-v3.0',
                  input_type='search_document').embeddings

search_index = AnnoyIndex(np.array(embeds).shape[1], 'angular')
# Add all the vectors to the search index
for i in range(len(embeds)):
    search_index.add_item(i, embeds[i])
search_index.build(10)  # 10 trees


def search(paragraphs):
  query_embed = co.embed(texts=paragraphs,
                         model='embed-english-v3.0',
                         input_type='search_query').embeddings

  results = {}

  for embed in query_embed:
    nearest = search_index.get_nns_by_vector(
      embed,
      1,
      include_distances=True)
    index = nearest[0][0]
    distance = nearest[1][0]

    if index not in results or distance < results[index]:
      results[index] = distance

  sorted_results = sorted(results.items(), key=lambda item: item[1])

  similar_item_indices, distances = zip(*sorted_results)

  rows = source_df.iloc[list(similar_item_indices)]

  query_results = pd.DataFrame(data={'claim': rows['claim'],
                                     'rating': rows['rating'],
                                     'source': rows['source'],
                                     'source_title': rows['source_title'],
                                     'source_link': rows['source_link'],
                                     'distance': distances})

  return query_results
