from typing import List,Tuple
import streamlit as st
import numpy as np
# import tensorflow_hub as hub
# import tensorflow as tf
from datetime import datetime
from itertools import compress
from sentence_transformers import SentenceTransformer
from scipy.stats import linregress, pearsonr
from gensim.models.coherencemodel import COHERENCE_MEASURES
import json

from query import (
    run_query,
    get_index_date_boundaries
)
from clustering import (
    cluster_vectors,
    compute_cluster_keywords,
    plot_cluster,
    compute_embedding_display_proj,
    build_topic_dataframes
)
from aspects import compute_aspect_similarities
from sentiment import plot_cluster_sentiment
from display_helpers import format_date_range

config_file = 'config.json'
config = json.load(open(config_file))

st.set_page_config(
    page_title=config['title'],
    page_icon="📊",
    layout="wide"
)

@st.cache(allow_output_mutation=True, max_entries=2)
def get_embedding_model(
        embedding_type: str
    ) -> SentenceTransformer:
    if embedding_type == "sbert":
        embedding_model = SentenceTransformer("all-MiniLM-L12-v2")
    elif embedding_type == "use_large":
        # prevent TensorFlow from allocating the entire GPU just to load the embedding model
        # gpus = tf.config.list_physical_devices("GPU")
        # for gpu in gpus:
        #     tf.config.experimental.set_memory_growth(gpu, True)
        # embedding_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
        print('no tf!')
    else:
        raise ValueError(f"Unsupported embedding type '{embedding_type}'.")
    return embedding_model

@st.cache(allow_output_mutation=True, max_entries=1)
def get_query_results(
        es_index:str, 
        embedding_type:str, 
        query:str, 
        date_range:Tuple[datetime,datetime],
        max_results:int, 
        sentiment_type:str, 
        use_responses:bool
    ) -> dict:
    embedding_model = get_embedding_model(embedding_type)
    query_results = run_query(
        config['es_uri'], 
        es_index, 
        embedding_type,
        embedding_model, 
        query, 
        date_range,
        sentiment_type,
        max_results=max_results,
        use_responses = use_responses)
    return query_results

@st.cache(allow_output_mutation=True)
def get_date_boundaries(
        es_index: str, 
        embedding_type: str, 
        use_responses: bool
    ) -> Tuple[datetime,datetime]:
    date_boundaries = get_index_date_boundaries(config['es_uri'], es_index, embedding_type, use_responses)
    return date_boundaries

@st.cache(allow_output_mutation=True, max_entries=1)
def get_aspect_similarities(
        tweet_embeddings: np.array, 
        embedding_type: str, 
        aspects: List[str]
    ):
    embedding_model = get_embedding_model(embedding_type)
    aspect_similarities = compute_aspect_similarities(
        tweet_embeddings, embedding_type, embedding_model, aspects)
    return aspect_similarities

@st.cache(allow_output_mutation=True, max_entries=1)
def get_cluster_assignments(*args, **kwargs):
    cluster_assignments = cluster_vectors(*args, **kwargs)
    return cluster_assignments

@st.cache(allow_output_mutation=True, max_entries=1)
def get_embedding_display_proj(*args, **kwargs):
    proj = compute_embedding_display_proj(*args, **kwargs)
    return proj

@st.cache(allow_output_mutation=True, max_entries=1)
def get_cluster_keywords(*args, **kwargs):
    keywords = compute_cluster_keywords(*args, **kwargs)
    return keywords

def run():
    # Step 1: Collect query, aspect, and clustering parameters
    with st.sidebar:
        st.title(config['title'])
        query_tab, aspects_tab, clustering_tab, sentiment_tab = st.tabs(["Query", "Aspects", "Clustering","Sentiment"])
        with query_tab:
            sorted_es_indices = sorted(config['es_indices'].keys())
            es_index = st.selectbox("Elasticsearch Index *", 
                                    sorted_es_indices,
                                    index = 0,
                                    key="elasticsearch_index")
            embedding_type = config['es_indices'][es_index]["embedding_type"]

            query = st.text_input("Find tweets similar to: *", 
                                  key="query", 
                                  value=config['es_indices'][es_index]["example_query"])
            use_responses = st.select_slider("Search by responses:", ["Off", "On"])
            use_responses = use_responses == 'On'

            filter_by_aspect = st.select_slider("Filter responses by aspect:", ["Off", "On"])
            filter_by_aspect = filter_by_aspect == 'On'
            
            date_boundaries = get_date_boundaries(es_index, embedding_type, use_responses)
            date_range = st.date_input("Date Range: *", key="date_range", value=date_boundaries,
                                       min_value=date_boundaries[0], max_value=date_boundaries[1])
            max_results = st.slider("Max Results *", 500, 10000, key="max_results", 
                                    value=5000, step=500)
            min_query_similarity = st.slider("Min Query Similarity", -1.0, 1.0, key="min_query_similarity", 
                                              value=0.25, step=0.05)
            st.markdown('<span style="color: red">*: changing causes query to re-run.</span>', unsafe_allow_html=True)

        with aspects_tab:
            aspect_defaults = config['es_indices'][es_index]["example_aspects"]
            aspects = [st.text_input(f"Aspect {i+1}:", key=f"aspect_{i+1}", value=aspect_defaults[i]) 
                       for i in range(2)]
            min_aspect_similarity = st.slider("Min Aspect Similarity", -1.0, 1.0, key="min_aspect_similarity", 
                                              value=0.25, step=0.05)
            show_trendline = st.checkbox("Show Trendline", key="show_trendline")

        with clustering_tab:
            # Clustering space & dimension reduction
            clustering_space = st.selectbox("Clustering Space", ["embedding","aspect"], key="clustering_space")
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
                kmeans_n_clusters = st.slider("# of Clusters (set 0 to detect)", 0, 30, key="kmeans_n_clusters", 
                                              value=5, step=1)
            else:
                hdbscan_min_cluster_size = st.slider("Min Cluster Size", 5, 100, key="hdbscan_min_cluster_size", 
                                                     value=5, step=5)
                hdbscan_min_samples = st.slider("Min Samples", 1, 100, key="hdbscan_min_samples",
                                                value=1, step=1)

            # Topic modeling settings
            num_topic_keywords = st.slider("# of keywords per cluster (topic)", 1, 20, key="num_topic_keywords", value=10, step=1)
            coherence_metrics = st.multiselect(
                "Topic Coherence Metrics", list(COHERENCE_MEASURES), default=["u_mass", "c_w2v"], key="coherence_metrics"
            )
        
        with sentiment_tab:
            sentiment_type = st.selectbox('Sentiment Type',['roberta','vader'])

            
    # Step 2: Execute the query and compute aspect similarities
    # (results are cached for unchanged query and aspect parameters)
    if not date_range:
        date_range = date_boundaries
    elif len(date_range) == 1:
        date_range = (date_range[0], date_boundaries[1])
    tweet_text, tweet_text_display, tweet_embeddings, tweet_scores, tweet_sentiments, timestamp = get_query_results(
        es_index, embedding_type, query, date_range, max_results,sentiment_type, use_responses
    )
    aspect_similarities = get_aspect_similarities(tweet_embeddings, embedding_type, aspects)

    # Step 3: Filter results by min query and aspect similarity
    min_query_similarity_filter = tweet_scores >= min_query_similarity
    min_aspect_similarity_filter = (aspect_similarities >= min_aspect_similarity).any(axis=-1)
    if filter_by_aspect:
        combined_filter = min_query_similarity_filter & min_aspect_similarity_filter
    else:
        combined_filter = min_query_similarity_filter
    
    filtered_aspect_similarities = aspect_similarities[combined_filter]
    filtered_tweet_embeddings = tweet_embeddings[combined_filter]
    filtered_tweet_text = list(compress(tweet_text, combined_filter))
    filtered_tweet_text_display = list(compress(tweet_text_display, combined_filter))
    filtered_timestamp = list(compress(timestamp,combined_filter))
    filtered_tweet_sentiments = tweet_sentiments[combined_filter]

    # filtered_tweet_sentiment = 

    print(f'after filtering, we get {len(filtered_tweet_text)} tweets..')
    
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
        print(f'we got {actual_n_clusters} clusters')

    if clustering_space == "embedding":
        print(f'we are resizing the vector with shape: {vectors_to_cluster.shape}')
        vectors_to_cluster = get_embedding_display_proj(vectors_to_cluster)

    # Step 5: Run topic modeling on clusters with tf-idf
    cluster_keywords = None
    if len(cluster_assignments) > 0:
        filtered_tweet_responses = [tweet_text[use_responses] for tweet_text in filtered_tweet_text]
        
        print('we are gonna cluster thes: ', filtered_tweet_responses[:10])

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
        # results_plot = 
        st.plotly_chart(
            plot_cluster(
                filtered_tweet_text_display,
                clustering_space,
                aspects,
                cluster_assignments,
                vectors_to_cluster,
                show_trendline,
                linreg_slope = linreg.slope if linreg else None,
                linreg_intercept = linreg.intercept if linreg else None,
            ), 
            use_container_width=True
        )

    # Step 8: Draw sentiment

    with st.expander(f"Results ({n_results} responses sentiment)", expanded=True):
        st.markdown('**Overall sentiment**')
        st.plotly_chart(
            plot_cluster_sentiment(cluster_assignments,filtered_tweet_sentiments,filtered_timestamp,None),
            use_container_width=True
        )

        for cluster_id in range(actual_n_clusters):
            st.markdown(f'**Cluster {cluster_id+1} sentiment**')
            st.plotly_chart(
                plot_cluster_sentiment(
                    cluster_assignments,
                    filtered_tweet_sentiments,
                    filtered_timestamp,
                    cluster_id,
                ),
                use_container_width=True
            )

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