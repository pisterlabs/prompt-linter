import numpy as np
import umap
import matplotlib.pyplot as plt
from datetime import datetime
from hdbscan import HDBSCAN
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search
from textwrap import wrap
from gensim import corpora
from gensim.utils import simple_preprocess
from gensim.models.coherencemodel import CoherenceModel

import cluster_helpers 

def text_wrap(text):
    return "<br>".join(wrap(text, width=80))

def get_tweet_text(hit):
    text = (hit["extended_tweet"]["full_text"] if "extended_tweet" in hit 
            else hit["full_text"] if "full_text" in hit 
            else hit["text"])
    quoted_text = None
    if "quoted_status" in hit:
        quoted_status = hit["quoted_status"]
        quoted_text = (quoted_status["extended_tweet"]["full_text"] if "extended_tweet" in quoted_status 
                      else quoted_status["full_text"] if "full_text" in quoted_status 
                      else quoted_status["text"])

    return text, quoted_text

def get_base_filters(embedding_type):
    return [{
        "exists": {
            "field": f"embedding.{embedding_type}.quoted"
        }
    }, {
        "exists": {
            "field": f"embedding.{embedding_type}.primary"
        }
    }]

def get_query(embedding_type, query_embedding, date_range):
    additional_filters = []
    if len(date_range) > 0:
        additional_filters.append({
            "range": {
                "created_at": {
                    "format": "strict_date",
                    "time_zone": "+00:00",
                    "gte": date_range[0].strftime("%Y-%m-%d")
                }
            }
        })
        if len(date_range) > 1:
            additional_filters[-1]["range"]["created_at"]["lte"] = date_range[1].strftime("%Y-%m-%d")

    query = {
        "_source": ["id_str", "text", "extended_tweet.full_text", "quoted_status.text", 
                    "quoted_status.extended_tweet.full_text", f"embedding.{embedding_type}.primary"],
        "query": {
            "script_score": {
                "query": {
                    "bool": {
                        "filter": get_base_filters(embedding_type) + additional_filters
                    }
                },
                "script": {
                    "source": f"dotProduct(params.query_vector, 'embedding.{embedding_type}.quoted') + 1.0",
                    "params": {"query_vector": query_embedding.tolist()}
                }
            }
        }
    }
    return query

def run_query(es_uri, es_index, embedding_type, embedding_model, query, date_range, max_results=1000):
    # Embed query
    if embedding_type == "sbert":
        query_embedding = embedding_model.encode(query, normalize_embeddings=True)
    elif embedding_type == "use_large":
        query_embedding = embedding_model([query]).numpy()[0]
    else:
        raise ValueError(f"Unsupported embedding type '{embedding_type}'.")

    # Use query embeddings to get responses to similar tweets
    with Elasticsearch(hosts=[es_uri], timeout=60, verify_certs=False) as es:
        s = Search(using=es, index=es_index)
        s = s.params(size=max_results)
        s.update_from_dict(get_query(embedding_type, query_embedding, date_range))

        tweet_text = []
        tweet_text_display = []
        tweet_embeddings = []
        tweet_scores = []
        for hit in s.execute():
            tweet_embeddings.append(np.array(hit["embedding"][embedding_type]["primary"]))
            text, quoted_text = get_tweet_text(hit)
            tweet_text.append((quoted_text, text))
            tweet_text_display.append(f"Tweet:<br>----------<br>{text_wrap(quoted_text)}<br><br>"
                                      f"Response:<br>----------<br>{text_wrap(text)}")
            tweet_scores.append(hit.meta.score-1.0)
            if len(tweet_embeddings) == max_results:
                break

        tweet_embeddings = np.vstack(tweet_embeddings)
        tweet_scores = np.array(tweet_scores)

    return tweet_text, tweet_text_display, tweet_embeddings, tweet_scores

def get_index_date_boundaries(es_uri, es_index, embedding_type):
    with Elasticsearch(hosts=[es_uri], timeout=60, verify_certs=False) as es:
        s = Search(using=es, index=es_index)
        s = s.params(size=0)
        s.update_from_dict({
            "query": {
                "bool": {"filter": get_base_filters(embedding_type)}
            },
            "aggs": {
                "min_date": {"min": {"field": "created_at", "format": "strict_date"}},
                "max_date": {"max": {"field": "created_at", "format": "strict_date"}}
            }
        })
        results = s.execute()
    min_date = datetime.strptime(results.aggregations.min_date.value_as_string, "%Y-%m-%d").date()
    max_date = datetime.strptime(results.aggregations.max_date.value_as_string, "%Y-%m-%d").date()
    return min_date, max_date

def compute_aspect_similarities(tweet_embeddings, embedding_type, embedding_model, aspects):
    # Embed aspects
    if embedding_type == "sbert":
        aspect_embeddings = embedding_model.encode(aspects, normalize_embeddings=True)
    elif embedding_type == "use_large":
        aspect_embeddings = embedding_model(aspects).numpy()
    else:
        raise ValueError(f"Unsupported embedding type '{embedding_type}'.")

    # Compute aspect similarity vector for each response.
    # Matrix multiplication will give cosine similarities
    # since all embeddings are normalized to unit sphere.
    aspect_similarities = tweet_embeddings @ aspect_embeddings.T

    return aspect_similarities

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
                kmeans = cluster_helpers.detect_optimal_clusters(vectors, plot_elbow=True)
                elbow_plot = plt.gcf()
            cluster_assignments = kmeans.predict(vectors)
    elif clustering_type == "hdbscan":
        hdbscan = HDBSCAN(min_cluster_size=hdbscan_min_cluster_size,
                          min_samples=hdbscan_min_samples)
        cluster_assignments = hdbscan.fit_predict(vectors)
    else:
        raise ValueError(f"Unsupported clustering type '{clustering_type}'.")

    silhouette_score = cluster_helpers.get_silhouette_score(vectors, cluster_assignments)
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
    tfidf_vocab = np.array(tfidf.get_feature_names())

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
    