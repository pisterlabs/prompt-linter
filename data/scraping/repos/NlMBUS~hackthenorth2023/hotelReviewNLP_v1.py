from sentence_transformers import SentenceTransformer
import cohere
from tqdm import tqdm
import time
import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import umap
import plotly.express as px
from sklearn import preprocessing


from torch.utils.tensorboard import SummaryWriter


#device = 'cuda' # nvidia-gpu
# device = 'mps' # apple-gpu
device = 'cpu' # no gpu

#co = cohere.Client("<API KEY>") # or None if you dont want to use Cohere
co = None
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

def enhanced(x):
    data = ""
    for key, item in x.items():
        data += f"{key}: {item} \n"
    return data

df = pd.read_csv('testData2.csv')

df_2 = df[['Review', 'Rating']].apply(enhanced, axis=1)

embeddings = encode(df_2.to_list())

print(embeddings) 


writer = SummaryWriter('review-hotel-data')
writer.add_embedding(embeddings[:3000],
                    list(zip(df['Review'][:3000].to_list(), df['Rating'][:3000].to_list())),
                    metadata_header=['Review', 'Rating'])
writer.close()


# reducer = umap.UMAP(n_neighbors=50, metric='cosine', verbose=True)
# reduced = reducer.fit_transform(embeddings)

df_3 = preprocessing.OrdinalEncoder().fit_transform(df[['Rating']]) # Map to unique numbers

for i in (0, 0.1, 0.2, 0.3, 0.4):
  for n in (2, 5, 10, 20, 50, 100, 200):
    print(f"images/weight_{i}_n_neighbors_{n}.png")
    reducer = umap.UMAP(n_neighbors=n, metric='cosine', target_weight=i, verbose=True)
    reduced = reducer.fit_transform(embeddings, df['Rating'])


    fig = px.scatter(x=reduced[:, 0],
                y=reduced[:, 1],
                hover_data=[df['Rating']],
                width=1000,
                height=1000,
                title=f"target_weight={i}")
    fig.update_traces(marker={'size': 2})
    fig.write_image(f"./images/weight_{i}_n_neighbors_{n}.png")







# text = ['I love cats!', 'I love dogs!', 'How do I fly from Waterloo to New York?']
# encodings = encode(text)

# print(encodings)