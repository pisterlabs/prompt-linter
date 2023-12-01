__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import types
import torch
import torch.nn
import pandas as pd
from openai import OpenAI
import inspect
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import tiktoken
from onnx2torch import convert
import chromadb

#chroma_client = chromadb.Client()
chroma_client = chromadb.PersistentClient(path="./embedding-db")
collection = chroma_client.get_or_create_collection(name="torch.nn")

st.set_page_config(page_title="PyTorch API", page_icon="📚")
st.markdown("# PyTorch API Reference")

whole_api_list = []

def get_api_list(pkg, pkg_name):
    frame = []
    for api in dir(pkg):
        full_name = f"{pkg_name}.{api}"
        if type(full_name) == types.ModuleType:
            pass # colorize the item.
        whole_api_list.append(full_name)
        frame.append([full_name, str(type(eval(full_name)))])
    return pd.DataFrame(frame, columns=("API Name", "Type"))

package_list = ['torch', 'torch.nn', 'torch.nn.functional']

for index, tab in enumerate(st.tabs(package_list)):
    with tab:
        pkg = package_list[index]
        st.write(f"Package: {pkg}")
        st.dataframe(get_api_list(eval(pkg), pkg))

# Function to retrieve the source code of a given function or class
def get_source_code(api_name):
    try:
        # Import the module and get the attribute (function/class)
        api = eval(api_name)

        # Get the source code
        source_code = inspect.getsource(api)
        return source_code
    except (ImportError, AttributeError):
        return f"API '{api_name}' not found."
    except TypeError:
        return f"Source code for '{api_name}' is not available (might be a built-in or C-extension)."

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(string))

def calculate_embedding(text, model="text-embedding-ada-002"):
    client = OpenAI(api_key=api_key)
    text = text.replace("\n", " ")
    return client.embeddings.create(input = [text], model=model).data[0].embedding

def visualize_2d(embs):
    pass

def visualize_3d(embs, api_list):
    pca = PCA(n_components=3)
    vis_dims = pca.fit_transform(embs)
    sub_matrix = np.array(vis_dims.tolist())
    
    kmeans_model = KMeans(n_clusters=20, n_init=10) # from 8 to 20
    kmeans_model.fit(sub_matrix)
    cluster_list = kmeans_model.labels_
    
    df = pd.DataFrame({'x': sub_matrix[:, 0],
                       'y': sub_matrix[:, 1],
                       'z': sub_matrix[:, 2],
                       'api_name': api_list,
                       'cluster': cluster_list,
                       })
    fig = px.scatter_3d(df, x='x', y='y', z='z', color='cluster', hover_name='api_name')
    fig.update_layout(width=900, height=900)
    fig.update_traces(textposition='middle center')
    fig.update_layout(uniformtext_minsize=12, uniformtext_mode='hide')
    def set_bgcolor(bg_color = "rgb(211, 211, 211)",
                grid_color="rgb(150, 150, 150)", 
                zeroline=False):
        return dict(showbackground=True,
                backgroundcolor=bg_color,
                gridcolor=grid_color,
                zeroline=zeroline)
    fig.update_scenes(xaxis=set_bgcolor(), 
                  yaxis=set_bgcolor(), 
                  zaxis=set_bgcolor())
    st.plotly_chart(fig, theme=None)

api_key  = st.text_input('Enter an OpenAI Key', None, type="password")
api_name = st.text_input('Enter an API', 'torch.nn.GRUCell')
if api_name not in whole_api_list:
    st.error('Bad API name!')
source_code = get_source_code(api_name)
st.code(source_code, language='python', line_numbers=True)
st.write(f"There are {num_tokens_from_string(source_code, 'cl100k_base')} tokens.")

def show_embedding():
    embedding = collection.get(ids=[api_name], include=['embeddings'])['embeddings']
    if embedding == []:
        if api_key != None:
            embedding = calculate_embedding(source_code, model='text-embedding-ada-002')
            collection.add(
                embeddings=[embedding,],
                documents=[source_code,],
                metadatas=[{"source": api_name},],
                ids=[api_name,]
            )
        else:
            st.session_state.need_openai_key = True
    if embedding != []:
        st.session_state.embedding = embedding

st.button('Get Embedding!', on_click=show_embedding)
if 'need_openai_key' in st.session_state:
    st.error("Please enter an OpenAI key.")
if 'embedding' in st.session_state:
    st.text_area(f"Output Embedding (Dim: {len(st.session_state.embedding)})", value=st.session_state.embedding, height=300, max_chars=None)


if st.button('Visualize Embedding!'):
    apis = collection.get()['ids']
    embs = np.array(collection.get(include=['embeddings'],)['embeddings'])
    visualize_2d(embs)
    visualize_3d(embs, apis)
# ['ada_embedding'] = df.combined.apply(lambda x: get_embedding(x, model='text-embedding-ada-002'))
# df.to_csv('output/embedded_1k_reviews.csv', index=False)

