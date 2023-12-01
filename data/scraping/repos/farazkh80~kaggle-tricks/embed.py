import pandas as pd
import regex as re
import os

import cohere 
import openai
import umap

import plotly.express as px
import streamlit as st

@st.cache_data
def extract_list_items(file_name):
    with open(file_name, "r") as f:
        lines = f.readlines()
    
    # extract all lines starting with a number
    lines = [line for line in lines if re.match(r"^\d+\.", line)]
    return lines

@st.cache_data
def embed(text):
    return co.embed(texts=text, model='embed-english-v2.0').embeddings


@st.cache_data
def reduce(embeds):
    reducer = umap.UMAP(n_neighbors=20) 
    return reducer.fit_transform(embeds)

@st.cache_data
def gen_summaries(texts):
    prompt = f"""
    # Task
    Given the list of items below, summarize the list of items in 3-4 bullet points, return only the bullet points.

    # List of items

    """
    summaries = []
    for text in texts:
        res = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt+text},
            ],
            max_tokens=500
        )
        summaries.append(res.choices[0].message.content)
    return summaries

@st.cache_data
def gen_keywords(texts):
    prompt = f"""
    # Task
    Given the list of items below, generate a comma seperated list of 3 keywords that best describe the list of items. ONLY RETURN THE KEYWORDS

    # Example List of items
    1. I love eating fruits in the morning
    2. Mornings, i start my day by eating two fruits at least
    3. fruits in the morning is the best way to start the day

    # Example Keywords
    fruits, morning, day

    # List of items

    """
    output = f"""

    # Example Keywords

    """
    keywords = []
    for text in texts:
        res = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt+text+output},
            ],
            max_tokens=500
        )
        keywords.append(res.choices[0].message.content)
    return keywords


st.title("""Kaggle Tips and Tricks""")

# build a list of line, file_name for all files under tricks folder
tips_tricks = []
for file_name in os.listdir("tricks"):
    file_name = os.path.join("tricks", file_name)
    tricks = extract_list_items(file_name)
    tips_tricks.extend([{'trick': t, 'file': file_name} for t in tricks])

tips_tricks_df = pd.DataFrame(tips_tricks, columns=['trick', 'file'])
co = cohere.Client(os.environ['CO_API_KEY'])

# embed the tips 
with st.spinner("Embedding tips..."):
    embeds = embed(tips_tricks_df['trick'].tolist())

# reduce the dimensionality of the embeddings
with st.spinner("Reducing dimensionality..."):
    umap_embeds = reduce(embeds)
    tips_tricks_df['x'] = umap_embeds[:,0]
    tips_tricks_df['y'] = umap_embeds[:,1]

# now cluster using k-means
k = st.slider("Number of clusters", 2, 10, 6)
from sklearn.cluster import KMeans
with st.spinner("Clustering tips..."):
    kmeans = KMeans(n_clusters=k, random_state=0).fit(umap_embeds)
    tips_tricks_df['cluster'] = kmeans.labels_

from textwrap import wrap
tips_tricks_df['trick'] = tips_tricks_df['trick'].apply(lambda x: "<br>".join(wrap(x, width=50)))

# group by cluster and concat all tricks
with st.spinner("Generating summaries..."):
    cluster_groups = tips_tricks_df.groupby('cluster').agg({'trick': lambda x: "\n".join(x)}).reset_index()
    cluster_groups['summary'] = gen_summaries(cluster_groups['trick'].tolist())
    cluster_groups['keywords'] = gen_keywords(cluster_groups['trick'].tolist())
    tips_tricks_df['trick_cluster_summary'] = cluster_groups['summary'][tips_tricks_df['cluster']].values
    tips_tricks_df['trick_cluster_keywords'] = cluster_groups['keywords'][tips_tricks_df['cluster']].values


tips_tricks_df['color'] = tips_tricks_df['trick_cluster_keywords'].astype(str)
c_fig = px.scatter(tips_tricks_df, x="x", y="y", color="color",
                   hover_data={
                          'x': False,
                          'y': False,
                          'color': False,
                          'trick': True,
                          'file': False,
                          'cluster': False,
                          'trick_cluster_summary': False
                          })

st.plotly_chart(c_fig, use_container_width=True, theme="streamlit")

st.write("## Summaries")
for i, row in cluster_groups.iterrows():
    with st.expander(row['keywords']):
        st.markdown(row['summary'])
