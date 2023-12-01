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
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import bentoml.sklearn
import json
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline

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

df_2 = df[['Review']].apply(enhanced, axis=1)

embeddings = encode(df_2.to_list())

print(embeddings) 


writer = SummaryWriter('review-hotel-data')
writer.add_embedding(embeddings[:3000],
                    list(zip(df['Review'][:3000].to_list(), df['Rating'][:3000].to_list())),
                    metadata_header=['Review', 'Rating'])
writer.close()


# reducer = umap.UMAP(n_neighbors=50, metric='cosine', verbose=True)
# reduced = reducer.fit_transform(embeddings)

df["Rating_id"] = preprocessing.OrdinalEncoder().fit_transform(df[['Rating']]) # Map to unique numbers

# for i in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
#   for n in (2, 5, 10, 20, 50, 100, 200):
reducer = umap.UMAP(n_neighbors=20, metric='cosine', target_weight=0.5, verbose=True)
reduced = reducer.fit_transform(embeddings, df["Rating_id"])

# Cluster Part:
clusters_model = KMeans(n_clusters=5, n_init='auto')

clusters = clusters_model.fit_predict(reduced)

pipeline = Pipeline([
    ('encode', encode),
    ('kmeans', clusters_model)
])




df["clusters"] = clusters
df["clusters"] = df["clusters"].astype(str)

writer = SummaryWriter('movie-data-clusters')
writer.add_embedding(embeddings[:3000], list(zip(df['Review'][:3000].to_list(), df['Rating'][:3000].to_list())), metadata_header=['Review', 'Rating'])
writer.close()


dataToWrite = {
  "x" : reduced[:, 0].tolist(),
  "y" : reduced[:, 1].tolist(),
  "rating" : df["Rating"].to_list()
}

df_23 = pd.DataFrame.from_dict(dataToWrite)

df_23.to_csv("currentData.csv")

# f = open("currentData.json", "w")

# f.write(str(dataToWrite))


fig = px.scatter(x=reduced[:, 0],
                y=reduced[:, 1],
                color=df["clusters"],
                hover_data=[df['Rating']],
                width=1000,
                height=1000)
fig.update_traces(marker={'size': 2})
# fig.write_image(f"./imagesColor/weight_{i}_n_neighbors_{n}.png")
fig.show()


print(df.head())
# embeddings_input = encode()

print(pipeline.predict(["Review: horrible customer service hotel stay february 3rd 4th 2007my friend picked hotel monaco appealing website online package included champagne late checkout 3 free valet gift spa weekend, friend checked room hours earlier came later, pulled valet young man just stood, asked valet open said, pull bags didn__Ç_é_ offer help, got garment bag suitcase came car key room number says not valet, car park car street pull, left key working asked valet park car gets, went room fine bottle champagne oil lotion gift spa"]))

# print(df["clusters"].predict("horrible customer service hotel stay february 3rd 4th 2007my friend picked hotel monaco appealing website online package included champagne late checkout 3 free valet gift spa weekend, friend checked room hours earlier came later, pulled valet young man just stood, asked valet open said, pull bags didn__Ç_é_ offer help, got garment bag suitcase came car key room number says not valet, car park car street pull, left key working asked valet park car gets, went room fine bottle champagne oil lotion gift spa, dressed went came got bed noticed blood drops pillows sheets pillows, disgusted just unbelievable, called desk sent somebody 20 minutes later, swapped sheets left apologizing, sunday morning called desk speak management sheets aggravated rude, apparently no manager kind supervisor weekend wait monday morning, young man spoke said cover food adding person changed sheets said fresh blood rude tone, checkout 3pm package booked, 12 1:30 staff maids tried walk room opening door apologizing closing, people called saying check 12 remind package, finally packed things went downstairs check, quickly signed paper took, way took closer look room, unfortunately covered food offered charged valet, called desk ask charges lady answered snapped saying aware problem experienced monday like told earlier, life treated like hotel, not sure hotel constantly problems lucky ones stay recommend anybody know,  "))

# print(df.head)



# def top_tfidf_words(cluster_data, n_words=10):
#     """
#     Extracts top n TF-IDF words for each cluster in the dataframe.

#     Parameters:
#     - cluster_data (pd.DataFrame): A dataframe with 'description' and 'cluster' columns.
#     - n_words (int): Number of top words to extract. Default is 10.

#     Returns:
#     - dict: A dictionary with cluster labels as keys and top n words as values.
#     """
#     # Initialize TF-IDF vectorizer
#     vectorizer = TfidfVectorizer(stop_words='english')

#     top_words = {}

#     # Extract top words for each cluster
#     for cluster in cluster_data['clusters'].unique():
#         descriptions = cluster_data[cluster_data['clusters'] == cluster]['description']
#         tfidf_matrix = vectorizer.fit_transform(descriptions)
#         feature_names = vectorizer.get_feature_names_out()

#         # Sum tf-idf weights for each word and get top n_words
#         summed_weights = tfidf_matrix.sum(axis=0).tolist()[0]
#         top_indices = sorted(range(len(summed_weights)), key=lambda i: summed_weights[i], reverse=True)[:n_words]
#         top_features = [feature_names[i] for i in top_indices]

#         top_words[cluster] = top_features

#     return top_words

# res = top_tfidf_words(df)
# for key, value in res.items():
#     print(f"{key}: {value}")