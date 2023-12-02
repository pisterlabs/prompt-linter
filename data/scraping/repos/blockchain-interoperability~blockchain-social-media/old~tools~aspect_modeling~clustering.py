from typing import List
from itertools import compress
import numpy as np
import pandas as pd
import umap
from hdbscan import HDBSCAN
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from kneed import KneeLocator

from tqdm import trange
from gensim import corpora
from gensim.utils import simple_preprocess
from gensim.models.coherencemodel import CoherenceModel

from sklearn.feature_extraction.text import TfidfVectorizer

import matplotlib.pyplot as plt
import plotly.graph_objects as go

def get_silhouette_score(embeddings, cluster_assignments):
    n_labels =  np.unique(cluster_assignments).shape[0]
    if n_labels > 1 and n_labels < embeddings.shape[0]:
        score = silhouette_score(embeddings, cluster_assignments)
    else:
        score = 0.
    return score

def detect_optimal_clusters(embeddings, k_range=(2,30), n_init=10, max_iter=300, 
                            random_state=None, show_progress=False, plot_elbow=False):
    # Determine actual k-range. The max can't be higher than the number of embeddings
    # and must also be greater than or equal to the min.
    min_k, max_k = k_range
    max_k = min(max_k, embeddings.shape[0])
    if max_k < min_k:
        min_k = max_k
    
    # Test all k in the k-range and gather the results
    kmeans_dict = {}
    for k in trange(min_k, max_k+1, 1, desc="KMeans n_clusters", disable=not show_progress):
        kmeans = KMeans(n_clusters=k, n_init=n_init, max_iter=max_iter, random_state=random_state)
        kmeans.fit(embeddings)
        kmeans_dict[k] = kmeans
    
    # Detect the elbow and optionally plot it, 
    # then select the KMeans instance with the optmal k
    x = np.arange(min_k, max_k+1)
    y = np.array([kmeans.inertia_ for _, kmeans in kmeans_dict.items()])
    kneedle = KneeLocator(x, y, curve="convex", direction="decreasing", 
                          online=True, interp_method="polynomial")
    if kneedle.elbow:
        optimal_k = round(kneedle.elbow)
        if plot_elbow:
            kneedle.plot_knee()
            #kneedle.plot_knee_normalized()
    else:
        #Fallback to KMeans default, or closest value in the k-range (this shouldn't really happen)
        print("Could not find optimal k using elbow method. Falling back to KMeans default.")
        optimal_k = 8 
        if optimal_k < min_k:
            optimal_k = min_k
        elif optimal_k > max_k:
            optimal_k = max_k
   
    return kmeans_dict[optimal_k]

def cluster_vectors(vectors, clustering_type, kmeans_n_clusters, 
                    hdbscan_min_cluster_size, hdbscan_min_samples, dreduce_dim):

    # before clustering do dimensionality reduction
    if dreduce_dim is not None and dreduce_dim < vectors.shape[-1]:
        dreduce_umap = umap.UMAP(n_neighbors=30, n_components=dreduce_dim, min_dist=0.0)
        vectors = dreduce_umap.fit_transform(vectors)
        vectors /= np.linalg.norm(vectors, axis=-1, keepdims=True)

    elbow_plot = None
    if clustering_type == "kmeans":
        if kmeans_n_clusters == 1 or vectors.shape[0] == 1:
            cluster_assignments = np.zeros(vectors.shape[0], dtype=np.int32)
        else:
            if kmeans_n_clusters > 0:
                kmeans_n_clusters = min(kmeans_n_clusters, vectors.shape[0])
                kmeans = KMeans(n_clusters=kmeans_n_clusters)
                kmeans.fit(vectors)
            else:
                kmeans = detect_optimal_clusters(vectors, plot_elbow=True)
                elbow_plot = plt.gcf()
            cluster_assignments = kmeans.predict(vectors)
    elif clustering_type == "hdbscan":
        hdbscan = HDBSCAN(min_cluster_size=hdbscan_min_cluster_size,
                          min_samples=hdbscan_min_samples)
        cluster_assignments = hdbscan.fit_predict(vectors)
    else:
        raise ValueError(f"Unsupported clustering type '{clustering_type}'.")

    silhouette_score = get_silhouette_score(vectors, cluster_assignments)
    return cluster_assignments, silhouette_score, elbow_plot

def compute_embedding_display_proj(embeddings):
    if embeddings.shape[-1] <= 2:
        return embeddings
    proj_umap = umap.UMAP(n_neighbors=10, n_components=2, min_dist=0.0)
    embedding_projections = proj_umap.fit_transform(embeddings)
    return embedding_projections

def compute_cluster_keywords(tweet_text, cluster_assignments, num_keywords, coherence_metrics):
    tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 1))
    tfidf_vectors = tfidf.fit_transform(tweet_text)
    
    # temporary fix for scikit-learn version difference
    # https://stackoverflow.com/questions/70215049/attributeerror-tfidfvectorizer-object-has-no-attribute-get-feature-names-out
    try:
        tfidf_vocab = np.array(tfidf.get_feature_names())
    except:
        tfidf_vocab = np.array(tfidf.get_feature_names_out())

    cluster_keywords = []
    cluster_tfidf_scores = []
    for i in np.unique(cluster_assignments):
        cluster_tfidf_vectors = tfidf_vectors[cluster_assignments == i]
        scores = np.mean(cluster_tfidf_vectors, axis=0).A[0]
        scores_sort_idx = np.flip(np.argsort(scores)[-num_keywords:])

        keywords = tfidf_vocab[scores_sort_idx].tolist()
        tfidf_scores = scores[scores_sort_idx].tolist()
        cluster_keywords.append(keywords)
        cluster_tfidf_scores.append(tfidf_scores)

    # compute topic coherence
    tweet_text_preprocessed = [simple_preprocess(t) for t in tweet_text]
    tweet_text_dictionary = corpora.Dictionary(tweet_text_preprocessed)
    cluster_coherence = {}
    for coherence_metric in coherence_metrics:
        cm = CoherenceModel(
            topics=cluster_keywords, texts=tweet_text_preprocessed, dictionary=tweet_text_dictionary, coherence=coherence_metric
        )
        cluster_coherence[coherence_metric] = cm.get_coherence_per_topic()

    return cluster_keywords, cluster_tfidf_scores, cluster_coherence


def plot_cluster(
        filtered_tweet_text_display: List[str],
        clustering_space: str,
        aspects: List[str],
        cluster_assignments: np.array,
        vectors_to_cluster: np.array,
        show_trendline: bool,
        linreg_slope: float = None,
        linreg_intercept: float = None,
    ):
    results_plot = go.Figure()
    results_plot.layout.margin = go.layout.Margin(b=0, l=0, r=0, t=30)
    if clustering_space == "aspect":
        results_plot.update_layout(xaxis_title=aspects[0], yaxis_title=aspects[1])
    for i in np.unique(cluster_assignments).tolist():
        cluster_vectors = vectors_to_cluster[cluster_assignments == i]
        cluster_tweet_text_display = list(compress(filtered_tweet_text_display, cluster_assignments == i))
        results_plot.add_trace(go.Scatter(x=cluster_vectors[:, 0],
                                            y=cluster_vectors[:, 1],
                                            mode="markers",
                                            marker=dict(color=i, colorscale="Viridis"),
                                            hoverinfo="text",
                                            hovertext=cluster_tweet_text_display,
                                            legendgroup=i,
                                            name=f"Cluster {i+1}"))
    if show_trendline and linreg_slope is not None:
        x = np.linspace(vectors_to_cluster[:, 0].min(), vectors_to_cluster[:, 0].max(), 2)
        y = linreg_slope * x + linreg_intercept
        results_plot.add_trace(go.Scatter(x=x, y=y, mode="lines", line={"dash": "longdash"}, showlegend=False))
    return results_plot

def build_topic_dataframes(cluster_assignments, cluster_keywords, cluster_tfidf_scores, cluster_coherence):
    topics_data = {}
    metrics_data = {"Cluster": [], "Avg_TF-IDF": []}
    for coherence_metric in cluster_coherence:
        metrics_data[f"Coherence_{coherence_metric}"] = []
        
    cluster_ids = np.unique(cluster_assignments)+1
    for i, cluster_id in enumerate(cluster_ids):
        topics_data[f"Cluster_{cluster_id}"] = cluster_keywords[i]
        topics_data[f"TF-IDF_{cluster_id}"] = cluster_tfidf_scores[i]
        metrics_data["Cluster"].append(f"Cluster_{cluster_id}")
        metrics_data["Avg_TF-IDF"].append(np.mean(cluster_tfidf_scores[i]))
        for coherence_metric in cluster_coherence:
            metrics_data[f"Coherence_{coherence_metric}"].append(cluster_coherence[coherence_metric][i])
    topics_df = pd.DataFrame.from_dict(topics_data)
    metrics_df = pd.DataFrame.from_dict(metrics_data)
    metrics_df = metrics_df.set_index("Cluster")
    metrics_df.loc["Avg."] = metrics_df.mean()
    return topics_df, metrics_df