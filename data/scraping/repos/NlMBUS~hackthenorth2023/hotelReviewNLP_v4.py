import os
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import cohere
import time
from tqdm import tqdm

df = pd.read_csv('testData2.csv')
data = []
for i in range(len(data)):
  data.append([df.loc[i][0], df.loc[i][1]])
  
#device = 'cuda' # nvidia-gpu
# device = 'mps' # apple-gpu
device = 'cpu' # no gpu

f = open("cohere-key.txt", "r")

co = cohere.Client(f.read()) # or None if you dont want to use Cohere
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2') # free offline transformer
def encode(text):
  if co is not None:
    if len(text) > 95:
      embed = []
      # prod key is 10000 per minute, free is 100. Cohere offers $300 in credits using htn2023
      sleep_time = 60 / 100
      k = 0
      start = time.time()
      for i in tqdm(range(0, len(text), 95)):
        embed += co.embed(texts=text[i:i + 95]).embeddings
        k += 1
        if k == 100:
          end = time.time()
          dur = end - start
          time.sleep(60 - dur if 60 - dur > 0 else 0)
          start = time.time()
          k = 0
    else:
      embed = co.embed(
          texts=text,
          model='embed-english-v2.0'
      ).embeddings
    embed = np.array(embed)
  else:
    embed = model.encode(text, device=device, show_progress_bar=True, batch_size=64)

  return embed




df = pd.DataFrame(data, columns = ['text', 'category'])

text = df['text']


vectors = encode(text)

vector_dimension = vectors.shape[0]
index = faiss.IndexFlatL2(vector_dimension)
faiss.normalize_L2(vectors)
index.add(vectors)

search_text = 'terrible'
search_vector = encode(search_text)
_vector = np.array([search_vector])
faiss.normalize_L2(_vector)

k = index.ntotal
distances, ann = index.search(_vector, k=k)

results = pd.DataFrame({'distances': distances[0], 'ann': ann[0]})
merge = pd.merge(results, df, left_on='ann', right_index=True)

labels = df['category']
category = labels[ann[0][0]]
print(labels, category)