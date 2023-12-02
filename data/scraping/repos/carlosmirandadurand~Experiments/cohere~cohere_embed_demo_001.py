#%%
# Cohere Demo: Embed Endpoint basic example 
# Source: https://docs.cohere.com/docs/embed-endpoint

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt

from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

import cohere



#%% 
# Connect

load_dotenv()
api_key = os.getenv('cohere_key__free_trial') 
co = cohere.Client(api_key)


#%% 
# Sample Training & Scoring Data

df = pd.read_csv("https://github.com/cohere-ai/notebooks/raw/main/notebooks/data/hello-world-kw.csv", names=["search_term"])
df.head(30)


#%% 
# Embed text by calling the Embed endpoint

def embed_text(texts):
    """
    Turns a piece of text into embeddings
    Arguments:
    text(str): the text to be turned into embeddings
    Returns:
    embedding(list): the embeddings
    """
    output = co.embed(
                model="embed-english-v2.0",
                texts=texts)
    embedding = output.embeddings

    return embedding


#%%
# Get embeddings of all search terms

df["search_term_embeds"] = embed_text(df["search_term"].tolist())

embeds = np.array(df["search_term_embeds"].tolist())

embeds



#%%
# Add a new query & get embeddings

new_query = "what is the history of hello world"

new_query_embeds = embed_text([new_query])[0]


#%% Calculate cosine similarity

def get_similarity(target, candidates):
    """
    Computes the similarity between a target text and a list of other texts
    Arguments:
    target(list[float]): the target text
    candidates(list[list[float]]): a list of other texts, or candidates
    Returns:
    sim(list[tuple]): candidate IDs and the similarity scores
    """
    # Turn list into array
    candidates = np.array(candidates)
    target = np.expand_dims(np.array(target),axis=0)

    # Calculate cosine similarity
    sim = cosine_similarity(target,candidates)
    sim = np.squeeze(sim).tolist()

    # Sort by descending order in similarity
    sim = list(enumerate(sim))
    sim = sorted(sim, key=lambda x:x[1], reverse=True)

    # Return similarity scores
    return sim


#%% 
# Get the similarity between the new query and existing queries

similarity = get_similarity(new_query_embeds, embeds)

# Display the top 5 FAQs
print("New query:")
print(new_query,'\n')

print("Similar queries:")
for idx,score in similarity[:25]:
  print(f"Similarity: {score:.2f};", df.iloc[idx]["search_term"])


#%%

from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
embeds_tsne = tsne.fit_transform(embeds)

df['x'] = embeds_tsne[:,0]
df['y'] = embeds_tsne[:,1]

#%%
# Create a scatterplot

plt.scatter(df['x'], df['y'])
plt.xlabel('First TSNE Dimension of the Embeddings')
plt.ylabel('Second TSNE Dimension')
plt.show()


# %%

# Plot the 2-dimension embeddings on a chart
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

result.interactive()


# %%
