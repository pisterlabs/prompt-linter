import cohere
import pandas as pd
import numpy as np
import altair as alt
import textwrap as tr
import umap
from sklearn.decomposition import PCA

api_key = 'cUkUMhISEr8QsUhZ8uaVMxZtdL3UJrlaESCyNtHR'
co = cohere.Client(api_key)

df_orig = pd.read_csv('https://raw.githubusercontent.com/cohere-ai/notebooks/main/notebooks/data/atis_intents_train.csv',names=['intent','query'])

sample_classes = ['atis_airfare', 'atis_airline', 'atis_ground_service']
df = df_orig.sample(frac=0.12, random_state=30)
df = df[df.intent.isin(sample_classes)]
df_orig = df_orig.drop(df.index)
df.reset_index(drop=True,inplace=True)

intents = df['intent'] #save for a later need
df.drop(columns=['intent'], inplace=True)
df.head()

def get_embeddings(texts,model='embed-english-v2.0'):
  output = co.embed(
                model=model,
                texts=texts)
  return output.embeddings

df['query_embeds'] = get_embeddings(df['query'].tolist())
df.head()

def get_pc(arr,n):
  pca = PCA(n_components=n)
  embeds_transform = pca.fit_transform(arr)
  return embeds_transform

embeds = np.array(df['query_embeds'].tolist())
embeds_pc = get_pc(embeds,10)

sample = 9

source = pd.DataFrame(embeds_pc)[:sample]
source = pd.concat([source,df['query']], axis=1)
source = source.melt(id_vars=['query'])

chart = alt.Chart(source).mark_rect().encode(
    x=alt.X('variable:N', title="Embedding"),
    y=alt.Y('query:N', title='',axis=alt.Axis(labelLimit=500)),
    color=alt.Color('value:Q', title="Value", scale=alt.Scale(
                range=["#917EF3", "#000000"]))
)

result = chart.configure(background='#ffffff'
        ).properties(
        width=700,
        height=400,
        title='Embeddings with 10 dimensions'
       ).configure_axis(
      labelFontSize=15,
      titleFontSize=12)

result.save('intuition-chart.html')