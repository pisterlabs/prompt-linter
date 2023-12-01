import cohere
import pandas as pd
import numpy as np
import altair as alt
import textwrap as tr
import umap

api_key = 'cUkUMhISEr8QsUhZ8uaVMxZtdL3UJrlaESCyNtHR'
co = cohere.Client(api_key)

df = pd.read_csv("https://github.com/cohere-ai/notebooks/raw/main/notebooks/data/hello-world-kw.csv", names=["search_term"])
df.head()

def embed_text(texts):
  output = co.embed(
                model="embed-english-v2.0",
                texts=texts)
  embedding = output.embeddings

  return embedding

df["search_term_embeds"] = embed_text(df["search_term"].tolist())
embeds = list(df["search_term_embeds"])

reducer = umap.UMAP(n_neighbors=49)
umap_embeds = reducer.fit_transform(embeds)

df['x'] = umap_embeds[:,0]
df['y'] = umap_embeds[:,1]

chart = alt.Chart(df).mark_circle(size=500).encode(
  x=
  alt.X('x',
      scale=alt.Scale(zero=False),
      axis=alt.Axis(labels=False, ticks=False, domain=False)
  ),

  y=
  alt.Y('y',
      scale=alt.Scale(zero=False),
      axis=alt.Axis(labels=False, ticks=False, domain=False)
  ),
  
  tooltip=['search_term']
  )

text = chart.mark_text(align='left', dx=15, size=12, color='black'
          ).encode(text='search_term', color= alt.value('black'))

result = (chart + text).configure(background="#FDF7F0"
      ).properties(
      width=1000,
      height=700,
      title="2D Embeddings"
      )

result.save('embed-chart.html')