"""
1. turn a list of urls into a list of embeddings
2. cluster the embeddings
"""
import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings import OpenAIEmbeddings
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import streamlit as st
import matplotlib.pyplot as plt

class DocumentClustering:
    def __init__(self, openai_embedding_model, openai_api_key, random_state):
        # Load environment variables
        load_dotenv()
        
        # Set attributes
        self.embedding_model = openai_embedding_model
        self.api_key = openai_api_key
        self.random_state = random_state

    def urls_to_embeddings(self, urls):
        """Fetches content from URLs and extracts embeddings."""
        loader = WebBaseLoader(urls)
        data = loader.load()
        embeddings_model = OpenAIEmbeddings(model=self.embedding_model, 
                                            openai_api_key=self.api_key)
        list_text = [i.page_content for i in data]
        return embeddings_model.embed_documents(list_text)

    def list_to_matrix(self, list_embeddings):
        """Converts list of embeddings to matrix."""
        return np.vstack(list_embeddings)

    def kmeans_cluster(self, matrix):
        """Performs KMeans clustering on the matrix."""
        kmeans = KMeans(n_clusters=4, init="k-means++", random_state=self.random_state)
        kmeans.fit(matrix)
        labels = kmeans.labels_
        return labels

    def tdistributed_neighbor_embedding_clustering_2(self, matrix):
        """Performs t-SNE dimensionality reduction."""
        tsne = TSNE(n_components=2, perplexity=15, random_state=self.random_state, 
                    init="random", learning_rate=200)
        vis_dims2 = tsne.fit_transform(matrix)
        return vis_dims2

    def visualize_clusters(self, vis_dims, labels):
        """Visualizes the clusters using t-SNE."""
        x = [x for x, y in vis_dims]
        y = [y for x, y in vis_dims]

        df = pd.DataFrame({"Cluster": labels})

        for category, color in enumerate(["purple", "green", "red", "blue"]):
            xs = np.array(x)[df.Cluster == category]
            ys = np.array(y)[df.Cluster == category]
            plt.scatter(xs, ys, color=color, alpha=0.3)

            avg_x = xs.mean()
            avg_y = ys.mean()

            plt.scatter(avg_x, avg_y, marker="x", color=color, s=100)
        plt.title("Clusters identified visualized in language 2d using t-SNE")
        plt.show()

# Usage
MY_SECRET_KEY = os.environ.get('MY_OPENAI_SECRET_KEY')
openai_embedding_model = "text-search-ada-doc-001"
random_state = 42

document_clustering = DocumentClustering(openai_embedding_model, MY_SECRET_KEY, random_state)
urls = ['','','','','']
list_of_embeddings = document_clustering.urls_to_embeddings(urls)
matrix_of_embeddings = document_clustering.list_to_matrix(list_of_embeddings)
labels = document_clustering.kmeans_cluster(matrix_of_embeddings)
vis_dims = document_clustering.tdistributed_neighbor_embedding_clustering_2(matrix_of_embeddings)
document_clustering.visualize_clusters(vis_dims, labels)


