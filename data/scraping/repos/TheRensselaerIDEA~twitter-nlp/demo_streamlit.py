import plotly.graph_objects as go
import streamlit as st
import numpy as np
import tensorflow_hub as hub
import tensorflow as tf
from itertools import compress
from sentence_transformers import SentenceTransformer
from scipy.stats import linregress, pearsonr
from gensim.models.coherencemodel import COHERENCE_MEASURES

import aspects as asp
from display_helpers import format_date_range, build_topic_dataframes

es_uri = "https://localhost:8080/elasticsearch/"
title = "Twitter Response Analysis"
st.set_page_config(
    page_title=title,
    page_icon="📊",
    layout="wide"
)

es_indices = {
    "coronavirus-data-all-lite": {
        "embedding_type": "use_large",
        "example_query": "Shut down the schools!",
        "example_aspects": ["child safety", "saving the economy"]
    },
    "coronavirus-data-masks": {
        "embedding_type": "use_large",
        "example_query": "Why should I wear a mask?",
        "example_aspects": ["protecting others", "personal freedom"]
    },
    "coronavirus-data-pubhealth-quotes": {
        "embedding_type": "sbert",
        "example_query": "Stay home, stay safe.",
        "example_aspects": ["CDC can't be trusted", "flatten the curve"]
    },
    "opioids-data-all": {
        "embedding_type": "use_large",
        "example_query": "Heroine",
        "example_aspects": ["can't stop", "stay away"]
    },
    "ukraine-data-lite": {
        "embedding_type": "sbert",
        "example_query": "US should arm Ukraine with fighter jets.",
        "example_aspects": ["risks of getting involved", "Russian war crimes"]
    },
    "ukraine-data-lite-oct22": {
        "embedding_type": "sbert",
        "example_query": "US should arm Ukraine with fighter jets.",
        "example_aspects": ["risks of getting involved", "Russian war crimes"]
    },
    "ukraine-data-lite-mar23": {
        "embedding_type": "sbert",
        "example_query": "US should arm Ukraine with fighter jets.",
        "example_aspects": ["risks of getting involved", "Russian war crimes"]
    },
    "vaccine-data-pubhealth-quotes": {
        "embedding_type": "sbert",
        "example_query": "Pregnant women should get vaccinated.",
        "example_aspects": ["risks to baby", "COVID while pregnant"]
    }
}

@st.cache_resource(max_entries=2)
def get_embedding_model(embedding_type):
    if embedding_type == "sbert":
        embedding_model = SentenceTransformer("all-MiniLM-L12-v2")
    elif embedding_type == "use_large":
        # prevent TensorFlow from allocating the entire GPU just to load the embedding model
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        embedding_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
    else:
        raise ValueError(f"Unsupported embedding type '{embedding_type}'.")
    return embedding_model

@st.cache_data(max_entries=1)
def get_query_results(es_index, embedding_type, query, date_range, max_results):
    embedding_model = get_embedding_model(embedding_type)
    query_results = asp.run_query(
        es_uri, 
        es_index, 
        embedding_type,
        embedding_model, 
        query, 
        date_range,
        max_results=max_results)
    return query_results

@st.cache_data
def get_date_boundaries(es_index, embedding_type):
    date_boundaries = asp.get_index_date_boundaries(es_uri, es_index, embedding_type)
    return date_boundaries

@st.cache_data(max_entries=1)
def get_aspect_similarities(tweet_embeddings, embedding_type, aspects):
    embedding_model = get_embedding_model(embedding_type)
    aspect_similarities = asp.compute_aspect_similarities(
        tweet_embeddings, embedding_type, embedding_model, aspects)
    return aspect_similarities

@st.cache_data(max_entries=1)
def get_cluster_assignments(*args, **kwargs):
    cluster_assignments = asp.cluster_vectors(*args, **kwargs)
    return cluster_assignments

@st.cache_data(max_entries=1)
def get_embedding_display_proj(*args, **kwargs):
    proj = asp.compute_embedding_display_proj(*args, **kwargs)
    return proj

@st.cache_data(max_entries=1)
def get_cluster_keywords(*args, **kwargs):
    keywords = asp.compute_cluster_keywords(*args, **kwargs)
    return keywords

def run():
    # Step 1: Collect query, aspect, and clustering parameters
    with st.sidebar:
        st.title(title)
        query_tab, aspects_tab, clustering_tab = st.tabs(["Query", "Aspects", "Clustering"])
        with query_tab:
            sorted_es_indices = sorted(es_indices.keys())
            es_index = st.selectbox("Elasticsearch Index *", 
                                    sorted_es_indices,
                                    index = sorted_es_indices.index("ukraine-data-lite-oct22"),
                                    key="elasticsearch_index")
            embedding_type = es_indices[es_index]["embedding_type"]

            query = st.text_input("Find responses to tweets similar to: *", 
                                  key="query", 
                                  value=es_indices[es_index]["example_query"])
            date_boundaries = get_date_boundaries(es_index, embedding_type)
            date_range = st.date_input("Date Range: *", key="date_range", value=date_boundaries,
                                       min_value=date_boundaries[0], max_value=date_boundaries[1])
            max_results = st.slider("Max Results *", 500, 10000, key="max_results", 
                                    value=5000, step=500)
            min_query_similarity = st.slider("Min Query Similarity", -1.0, 1.0, key="min_query_similarity", 
                                              value=0.25, step=0.05)
            st.markdown('<span style="color: red">*: changing causes query to re-run.</span>', unsafe_allow_html=True)

        with aspects_tab:
            aspect_defaults = es_indices[es_index]["example_aspects"]
            aspects = [st.text_input(f"Aspect {i+1}:", key=f"aspect_{i+1}", value=aspect_defaults[i]) 
                       for i in range(2)]
            min_aspect_similarity = st.slider("Min Aspect Similarity", -1.0, 1.0, key="min_aspect_similarity", 
                                              value=0.25, step=0.05)
            show_trendline = st.checkbox("Show Trendline", key="show_trendline")

        with clustering_tab:
            # Clustering space & dimension reduction
            clustering_space = st.selectbox("Clustering Space", ["aspect", "embedding"], key="clustering_space")
            dreduce_dim = None
            if clustering_space == "embedding":
                max_dim = 512 if embedding_type == "use_large" else 384
                dreduce_dim = st.slider("Reduce to dimension before clustering", 2, max_dim, key="dreduce_dim", value=max_dim, step=1)
            
            # Clustering type & hparams
            clustering_type = st.selectbox("Clustering Type", ["kmeans", "hdbscan"], key="clustering_type")
            kmeans_n_clusters = None
            hdbscan_min_cluster_size = None
            hdbscan_min_samples = None
            if clustering_type == "kmeans":
                kmeans_n_clusters = st.slider("Num. Clusters (set 0 to detect)", 0, 30, key="kmeans_n_clusters", 
                                              value=5, step=1)
            else:
                hdbscan_min_cluster_size = st.slider("Min Cluster Size", 5, 100, key="hdbscan_min_cluster_size", 
                                                     value=5, step=5)
                hdbscan_min_samples = st.slider("Min Samples", 1, 100, key="hdbscan_min_samples",
                                                value=1, step=1)

            # Topic modeling settings
            num_topic_keywords = st.slider("Num. keywords per cluster (topic)", 1, 20, key="num_topic_keywords", value=10, step=1)
            coherence_metrics = st.multiselect(
                "Topic Coherence Metrics", list(COHERENCE_MEASURES), default=["u_mass", "c_w2v"], key="coherence_metrics"
            )
            
    # Step 2: Execute the query and compute aspect similarities
    # (results are cached for unchanged query and aspect parameters)
    if not date_range:
        date_range = date_boundaries
    elif len(date_range) == 1:
        date_range = (date_range[0], date_boundaries[1])
    tweet_text, tweet_text_display, tweet_embeddings, tweet_scores = get_query_results(
        es_index, embedding_type, query, date_range, max_results
    )
    aspect_similarities = get_aspect_similarities(tweet_embeddings, embedding_type, aspects)

    # Step 3: Filter results by min query and aspect similarity
    min_query_similarity_filter = tweet_scores >= min_query_similarity
    min_aspect_similarity_filter = (aspect_similarities >= min_aspect_similarity).any(axis=-1)
    combined_filter = min_query_similarity_filter & min_aspect_similarity_filter
    
    filtered_aspect_similarities = aspect_similarities[combined_filter]
    filtered_tweet_embeddings = tweet_embeddings[combined_filter]
    filtered_tweet_text = list(compress(tweet_text, combined_filter))
    filtered_tweet_text_display = list(compress(tweet_text_display, combined_filter))
    
    # Step 4: Run clustering
    vectors_to_cluster = filtered_aspect_similarities if clustering_space == "aspect" else filtered_tweet_embeddings
    n_results = vectors_to_cluster.shape[0]
    cluster_assignments = []
    silhouette_score = 0.
    elbow_plot = None
    actual_n_clusters = 0
    
    if vectors_to_cluster.shape[0] > 0:
        cluster_assignments, silhouette_score, elbow_plot = get_cluster_assignments(
            vectors_to_cluster, clustering_type, kmeans_n_clusters, hdbscan_min_cluster_size, hdbscan_min_samples, dreduce_dim
        )
        actual_n_clusters = np.max(cluster_assignments) + 1

    if clustering_space == "embedding":
        vectors_to_cluster = get_embedding_display_proj(vectors_to_cluster)

    # Step 5: Run topic modeling on clusters with tf-idf
    cluster_keywords = None
    if len(cluster_assignments) > 0:
        filtered_tweet_responses = [tweet_text[1] for tweet_text in filtered_tweet_text]
        cluster_keywords, cluster_tfidf_scores, cluster_coherence = get_cluster_keywords(
            filtered_tweet_responses, cluster_assignments, num_topic_keywords, coherence_metrics
        )
    
    # Step 6: Run linear regression
    linreg = None
    if n_results > 0:
        linreg = linregress(x=vectors_to_cluster[:, 0],
                            y=vectors_to_cluster[:, 1])
    rvalue = linreg.rvalue if linreg is not None else 0.
    pvalue = linreg.pvalue if linreg is not None else 0.

    # Step 7: Display the results
    with st.expander(f"Results ({n_results} responses)", expanded=True):
        # display query / clustering results
        st.markdown(f"**Index:** {es_index}; &nbsp;&nbsp; "
                    f"**Query:** \"{query}\" ({format_date_range(date_range)}); &nbsp;&nbsp; "
                    f"**Pearson's r:** {rvalue:.3f} (p={pvalue:.3f}); &nbsp;&nbsp; "
                    f"**Clusters:** {actual_n_clusters}; &nbsp;&nbsp; "
                    f"**Silhouette:** {silhouette_score:.3f}", unsafe_allow_html=True)
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
        if show_trendline and linreg is not None:
            x = np.linspace(vectors_to_cluster[:, 0].min(), vectors_to_cluster[:, 0].max(), 2)
            y = linreg.slope * x + linreg.intercept
            results_plot.add_trace(go.Scatter(x=x, y=y, mode="lines", line={"dash": "longdash"}, showlegend=False))
        st.plotly_chart(results_plot, use_container_width=True)

    # display topic results
    if cluster_keywords is not None:
        topics_df, metrics_df = build_topic_dataframes(
            cluster_assignments, cluster_keywords, cluster_tfidf_scores, cluster_coherence
        )
        with st.expander(f"Results ({n_results} response topics)", expanded=True):
            st.write("Topics:")
            st.dataframe(topics_df)
            st.write("Metrics:")
            st.dataframe(metrics_df)

    # display k-means elbow plot
    if elbow_plot is not None:
        with st.expander("k-means Elbow plot", expanded=True):
            st.pyplot(elbow_plot)

if __name__ == "__main__":
    run()