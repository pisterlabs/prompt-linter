import streamlit as st
import os
import time
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)

import plotly.express as px
import plotly.graph_objects as go
import matplotlib.cm as cm
from streamlit_extras import stateful_button

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.spatial.distance import cdist
import numpy as np
import textwrap

from langchain.chat_models import ChatOpenAI
from langchain.schema.messages import HumanMessage, SystemMessage

from bertopic import BERTopic
from bertopic.dimensionality import BaseDimensionalityReduction
from umap import UMAP
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.representation import OpenAI


DATA_PATH = "./reddit_legal_cluster_test_results.parquet"
TEXT_COL_NAME = 'body'
EMBEDDING_COL_NAME = 'embeddings'

st.set_page_config(layout="wide")
st.title("Guided Clustering App")

# Load data
@st.cache_data
def load_data():
    df = pd.read_parquet(DATA_PATH)
    df = df.drop(columns=['embeddings'])
    return df

# Load embeddings
@st.cache_data
def load_embeddings():
    df = pd.read_parquet(DATA_PATH)
    return df['embeddings']

# Get gpt-4 instance
@st.cache_resource
def get_gpt4():
    return ChatOpenAI(temperature=0, model="gpt-4")

def convert_df(df):
    return df.to_csv().encode('utf-8')


def join_embeddings(text_df, embeddings_df):
    joined_df = text_df.join(embeddings_df)
    if EMBEDDING_COL_NAME in joined_df.columns:
        return joined_df
    else:
        raise ValueError(f"'{EMBEDDING_COL_NAME}' column not found in the joined dataframe.")
    
    
def fit_kmeans_and_get_metrics(df, embedding_col_name, min_clusters, max_clusters):
    metrics_list = []
    embeddings = np.vstack(df[embedding_col_name].values)

    for n_clusters in range(min_clusters, max_clusters+1):
        kmeans = KMeans(n_clusters=n_clusters, n_init='auto', random_state=0).fit(embeddings)
        labels = kmeans.labels_

        silhouette_avg = silhouette_score(embeddings, labels)
        davies_bouldin_avg = davies_bouldin_score(embeddings, labels)

        metrics_list.append({
            'n_clusters': n_clusters,
            'silhouette_score': silhouette_avg,
            'davies_bouldin_score': davies_bouldin_avg,
        })

    metrics_df = pd.DataFrame(metrics_list)
    return metrics_df


def plot_metrics(metrics_df):
    # Flip the sign so plot reads "higher is better" across metrics
    metrics_df["davies_bouldin_score"] = np.negative(metrics_df["davies_bouldin_score"])
    
    # Normalize the metrics to the range [0, 1]
    scaler = MinMaxScaler()
    metrics_df[['silhouette_score', 'davies_bouldin_score']] = scaler.fit_transform(metrics_df[['silhouette_score', 'davies_bouldin_score']])


    fig = go.Figure()

    # Add traces
    fig.add_trace(go.Scatter(x=metrics_df['n_clusters'], y=metrics_df['silhouette_score'],
                        mode='lines',
                        name='silhouette_score'))
    fig.add_trace(go.Scatter(x=metrics_df['n_clusters'], y=metrics_df['davies_bouldin_score'],
                        mode='lines',
                        name='davies_bouldin_score'))

    # Add layout details
    fig.update_layout(
        title="Clustering Metrics | Normalized Range and Sign",
        xaxis_title="Number of Clusters",
        yaxis_title="Normalized Metric Score",
        legend_title="Metric",
        autosize=False,
        width=800,
        height=500,
        yaxis=dict(
            tickmode = 'linear',
            tick0 = 0,
            dtick = 0.1
        )
    )
    return fig


def apply_tsne(df, embedding_col_name):
    tsne = TSNE(n_components=2, 
                metric='cosine',
                perplexity=10,
                random_state=42)
    tsne_results = tsne.fit_transform(np.vstack(df[embedding_col_name].values))
    df['tsne_x'] = tsne_results[:, 0]
    df['tsne_y'] = tsne_results[:, 1]
    return df


def apply_umap(df, embedding_col_name):
    umap_model = UMAP(
        n_neighbors=15, 
        n_components=2, 
        min_dist=0.1,
        angular_rp_forest=True,
        metric='cosine')
    umap_results = umap_model.fit_transform(np.vstack(df[embedding_col_name].values))
    df['umap_x'] = umap_results[:, 0]
    df['umap_y'] = umap_results[:, 1]
    return df


def plot_tsne_data(df, label_col, max_line_length=75):
    # Create an empty figure
    fig = go.Figure()

    # Get the viridis color map with an appropriate number of colors
    viridis = cm.get_cmap('viridis', df[label_col].nunique())

    # Add separate traces for each unique ClusterLabel
    for i, label in enumerate(sorted(df[label_col].unique())):
        df_label = df[df[label_col] == label]
        df_label[TEXT_COL_NAME] = df_label[TEXT_COL_NAME].apply(
            lambda x: '<br>'.join([x[i:i+max_line_length] for i in range(0, len(x), max_line_length)])
        )
        display_name = str(label) if len(str(label)) <= 50 else str(label)[:50] + "..."
        
        # Adjust marker aesthetics
        fig.add_trace(
            go.Scattergl(
                x=df_label["tsne_x"],
                y=df_label["tsne_y"],
                hovertext=df_label[TEXT_COL_NAME],
                hoverinfo='text',
                mode='markers',
                name=display_name,
                marker=dict(
                    color='rgb'+str(tuple(int(x*255) for x in viridis(i)[:3])),
                    size=4, 
                    opacity=0, 
                    )
            )
        )

    fig.update_layout(
        title=f"Coverage Question Clusters | {len(df[label_col].unique())} clusters. {len(df):,} cases<br><sup><i>OpenAI embeddings. GPT-3.5-turbo labels</i></sup>",
        title_font_size=26,
        hoverlabel=dict(
            bgcolor="rgba(255, 255, 255, 0.7)",  # semi-transparent white background
            font_size=14, 
            font_family="Roboto, sans-serif",
            font_color="black"  # black text
        ),
        xaxis=dict(title="X", showgrid=True, gridwidth=0.25, gridcolor='lightgrey'),
        yaxis=dict(title="Y", showgrid=True, gridwidth=0.25, gridcolor='lightgrey'),
        width=1200,
        height=800,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.025,  # Position the legend to the right of the plot
            itemsizing='constant',
            font=dict(size=14),
        ), 
    )

    # Update traces for a cleaner look
    fig.update_traces(marker=dict(size=7, opacity=0.95),
                      marker_line=dict(width=.1, color='darkgrey'),
                      textfont=dict(size=12, family="Roboto, sans-serif", color="lightgrey"),)

    return fig



def plot_umap_data(df, label_col, max_line_length=75):
    # Create an empty figure
    fig = go.Figure()

    # Get the viridis color map with an appropriate number of colors
    viridis = cm.get_cmap('viridis', df[label_col].nunique())

    # Add separate traces for each unique ClusterLabel
    for i, label in enumerate(sorted(df[label_col].unique())):
        df_label = df[df[label_col] == label]
        df_label[TEXT_COL_NAME] = df_label[TEXT_COL_NAME].apply(
            lambda x: '<br>'.join([x[i:i+max_line_length] for i in range(0, len(x), max_line_length)])
        )
        display_name = str(label) if len(str(label)) <= 50 else str(label)[:50] + "..."
        # Adjust outlier marker
        if label == "Outlier":
            color = 'rgb(128,128,128)'  # RGB for grey
            opacity = 0.25  # 50% opaque
            size = 2
        else:
            color = 'rgb'+str(tuple(int(x*255) for x in viridis(i)[:3]))
            opacity = 0
            size = 4
        # Adjust marker aesthetics
        fig.add_trace(
            go.Scattergl(
                x=df_label["umap_x"],
                y=df_label["umap_y"],
                hovertext=df_label[TEXT_COL_NAME],
                hoverinfo='text',
                mode='markers',
                name=display_name,
                marker=dict(
                    color=color,
                    size=size, 
                    opacity=opacity, 
                    )
            )
        )

    fig.update_layout(
        title=f"Coverage Question Clusters | {len(df[label_col].unique())} clusters. {len(df):,} cases<br><sup><i>OpenAI embeddings. GPT-3.5-turbo labels</i></sup>",
        title_font_size=26,
        hoverlabel=dict(
            bgcolor="rgba(255, 255, 255, 0.7)",  # semi-transparent white background
            font_size=14, 
            font_family="Roboto, sans-serif",
            font_color="black"  # black text
        ),
        xaxis=dict(title="X", showgrid=True, gridwidth=0.25, gridcolor='lightgrey'),
        yaxis=dict(title="Y", showgrid=True, gridwidth=0.25, gridcolor='lightgrey'),
        width=1200,
        height=800,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.025,  # Position the legend to the right of the plot
            itemsizing='constant',
            font=dict(size=14),
        ), 
    )

    # Update traces for a cleaner look
    fig.update_traces(marker=dict(size=5, opacity=0.95),
                      marker_line=dict(width=.15, color='darkgrey'),
                      textfont=dict(size=12, family="Roboto, sans-serif", color="lightgrey"),)

    return fig


def fit_bertopic_and_get_labels_hdbscan(df):
    df = df.copy()
    df = df.reset_index(drop=True)
    docs = list(df[TEXT_COL_NAME])
    embeds = np.vstack(df[EMBEDDING_COL_NAME].values)
    vectorizer_model = CountVectorizer(min_df=5, stop_words = 'english')
    umap_model = UMAP(
        n_neighbors=10, 
        n_components=5, 
        min_dist=0.001, 
        metric='cosine',
        angular_rp_forest=True, 
        random_state=42)
    representation_model = OpenAI(model="gpt-3.5-turbo-16k", chat=True, nr_docs=30, delay_in_seconds=2)
    
    topic_model = BERTopic(
        nr_topics='auto',
        umap_model=umap_model,
        vectorizer_model=vectorizer_model,
        representation_model=representation_model,
        verbose=False)
    
    topics, _ = topic_model.fit_transform(docs, embeds)
    df_topics = topic_model.get_document_info(docs)
    df_out = df.join(df_topics)
    df_out['Topic_Label'] = np.where(df_out['Topic']== -1, "Outlier", df_out['Top_n_words'])
    df_out['Topic_Label_'] = df_out['Topic_Label'].apply(lambda x: x if len(x) <= 35 else x[:35] + "...")
    topic_mapping = df_out.set_index('Topic')['Topic_Label_'].to_dict()
    topic_model.set_topic_labels(topic_mapping)
    
    # Create a DataFrame for Topic, Average Probability, and Cluster Persistence
    topic_avg_prob_df = df_out.groupby('Topic_Label')['Probability'].mean().reset_index(name='Average Probability')
    topic_counts_df = df_out.groupby('Topic_Label').size().reset_index(name='Count')
    
    df_cluster_info = pd.merge(topic_avg_prob_df, topic_counts_df, on='Topic_Label', how='left')
    df_cluster_info.sort_values(by="Count", ascending=False, inplace=True)
    
    return df_out, df_cluster_info, topic_model


def fit_bertopic_and_get_labels_kmeans(df, n_clusters):
    df = df.copy()
    df = df.reset_index(drop=True)
    docs = list(df[TEXT_COL_NAME])
    embeds = np.vstack(df[EMBEDDING_COL_NAME].values)
    vectorizer_model = CountVectorizer(min_df=5, stop_words = 'english')
    empty_dimensionality_model = BaseDimensionalityReduction()
    cluster_model = KMeans(n_clusters=n_clusters, n_init='auto', random_state=42)
    representation_model = OpenAI(model="gpt-3.5-turbo-16k", chat=True, nr_docs=30, delay_in_seconds=3)
    
    topic_model = BERTopic(
        hdbscan_model=cluster_model,
        umap_model=empty_dimensionality_model,
        vectorizer_model=vectorizer_model,
        representation_model=representation_model,
        verbose=False)
    topic_model.fit(docs, embeds)

    df_topics = topic_model.get_document_info(docs)
    df_out = df.join(df_topics)
    df_out['Topic_Label'] = df_out['Top_n_words'].apply(lambda x: x if len(x) <= 50 else x[:50] + "...")
    
    return df_out, topic_model, docs


def plot_topic_frequency_hdbscan(df, num_categories):
    # Filter out outliers
    df = df[df['Topic'] != -1]

    # Extract topic frequency from the dataframe
    topic_freq = df['Top_n_words'].value_counts().nlargest(num_categories).reset_index()
    topic_freq.columns = ['Top_n_words', 'Frequency']

    # Get the top n words for each topic
    top_n_words = df['Top_n_words'].values

    # Create a horizontal bar plot for frequency
    fig = go.Figure(data=go.Bar(
        y=[str(label) if len(str(label)) <= 50 else str(label)[:50] + "..." for label in top_n_words], 
        x=topic_freq['Frequency'], 
        orientation='h'))

    fig.update_layout(title_text='Topic Frequency', autosize=True, height=len(topic_freq)*20)

    return fig


def plot_topic_frequency_kmeans(df, num_categories):
    # Extract topic frequency from the dataframe
    topic_freq = df['Top_n_words'].value_counts().nlargest(num_categories).reset_index()
    topic_freq.columns = ['Top_n_words', 'Frequency']

    # Get the top n words for each topic
    top_n_words = df['Top_n_words'].values

    # Create a horizontal bar plot for frequency
    fig = go.Figure(data=go.Bar(y=top_n_words, x=topic_freq['Frequency'], orientation='h'))

    fig.update_layout(title_text='Topic Frequency', autosize=True, height=len(topic_freq)*20)

    return fig


def split_frame(input_df, rows):
    df = [input_df.loc[i : i + rows - 1, :] for i in range(0, len(input_df), rows)]
    return df


def paginate_df(name: str, dataset, streamlit_object: str, disabled=None, num_rows=None):
    top_menu = st.columns(2)
    sort_field = None
    sort_direction = None
    with top_menu[0]:
        sort_field = st.selectbox("Sort By", options=dataset.columns)
    with top_menu[1]:
        sort_direction = st.radio(
            "Direction", options=["⬆️", "⬇️"], horizontal=True
        )
    dataset = dataset.sort_values(
        by=sort_field, ascending=sort_direction == "⬆️", ignore_index=True
    )
    pagination = st.container()

    bottom_menu = st.columns((4, 1, 1))
    with bottom_menu[2]:
        batch_size = st.selectbox("Page Size", options=[25, 50, 100], key=f"{name}")
    with bottom_menu[1]:
        total_pages = (
            int(len(dataset) / batch_size) if int(len(dataset) / batch_size) > 0 else 1
        )
        current_page = st.number_input(
            "Page", min_value=1, max_value=total_pages, step=1
        )
    with bottom_menu[0]:
        st.markdown(f"Page **{current_page:,}** of **{total_pages:,}**")
        st.markdown(f"Total Records: **{len(dataset):,}**")

    pages = split_frame(dataset, batch_size)

    if streamlit_object == 'df':
        pagination.dataframe(data=pages[current_page - 1], hide_index=True, use_container_width=True)
    
    if streamlit_object == 'editable df':
        pagination.data_editor(data=pages[current_page - 1], hide_index=True, disabled=disabled, num_rows=num_rows, use_container_width=True)


def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns

    Args:
        df (pd.DataFrame): Original dataframe

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    modify = st.checkbox("Add filters")

    if not modify:
        return df

    df = df.copy()

    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col], format='%Y-%m-%d %H:%M:%S')
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    modification_container = st.container()

    with modification_container:
        to_filter_columns = st.multiselect("Filter dataframe on", df.columns)
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            # Treat columns with < 10 unique values as categorical
            if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                user_cat_input = right.multiselect(
                    f"Values for {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
                )
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Values for {column}",
                    min_value=_min,
                    max_value=_max,
                    value=(_min, _max),
                    step=step,
                )
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df[column]):
                user_date_input = right.date_input(
                    f"Values for {column}",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f"Substring or regex in {column}",
                )
                if user_text_input:
                    df = df[df[column].astype(str).str.contains(user_text_input)]

    return df


def main():
    load_dotenv()
    gpt4 = get_gpt4()
    df = load_data()
    st.session_state['df'] = df
    embeddings = load_embeddings()
    st.session_state['embeddings'] = embeddings      

    tab = st.sidebar.selectbox("Choose a tab", ["Filter Data", "K-Means Cluster Evaluation", "HDBSCAN Cluster Evaluation", "Clustering Results"])

    if tab == "Filter Data":
        st.markdown("___")
        with st.expander(label="Intro", expanded=False):
            st.markdown("""
    Welcome to the Guided Clustering app, designed to provide state-of-the-art clustering solutions in fast and easy way. This tool integrates advanced data processing techniques with the analytical power of GPT-4.
    
    #### Here's how you can leverage this tool:

    1. **Filter the Data:** Begin your analysis by selecting a dataset to explore. Choose filters and when you're happy click `Comfirm Selection`.

    2. **K-Means Clustering:** Head to the next tab to fit a K-Means model to your data. We'll guided you through an initial analysis, then we'll build a final model, generate topic labels with LibertyGPT, and visualize the results. 

    3. **HDBSCAN Clustering:** On the next tab we'll build another (and more advanced) model using "hierarchical density based clustering around noise". There are no parameters here, so just click the button to run the end-to-end analysis.

    4. **Wrapping Things Up:** If you're all set, navigate to the last tab to get a summary report along with an option to export the data. Or, run the app again on a different cut of the data.

                        """)
        st.markdown("___")
        
        if 'filtered_df' in st.session_state:
            dataset = filter_dataframe(st.session_state['filtered_df'])
        else:
            dataset = filter_dataframe(st.session_state['df'])
        paginate_df('Selected', dataset, 'df')
        if st.button("Confirm Selection"):
            st.session_state['filtered_df'] = dataset
        if st.button("Reset Filters"):
            st.session_state['filtered_df'] = st.session_state['df']

    # Main app functionality to guide users through a clustering analysis
    elif tab == "K-Means Cluster Evaluation":
        st.markdown("___")
        with st.expander(label="Intro", expanded=False):
            st.markdown("""    
    #### Here's how you can use this tab:

    1. **Select Cluster Range:** The first, and crucial part of K-Means, is figuring out how many true clusters exist. Let's begin by selecting a range of `n_clusters` to test. 10-25 is usually a good place to start. You can always come back to adjust and try again.

    2. **Review Plots and Feedback:** After the clustering process, you'll be presented with results along with feedback generated by GPT-4. From here, you can either decide on a number of clusters, or try another range.

    3. **Visualize Results:** We'll build a 'final' model based on your selection, and have LibertyGPT read ~30 samples per cluster to come up with labels.

    4. **Iterative Analysis:** The dynamic nature of this tool allows you to redo the analysis as needed. Adjust your cluster range, review the new outputs, or head back to the filtering tab to change the scope of your analysis.
    5. **That's it!:** When you're ready, head to the next tab where we'll compliment our partitioning-based K-Means model with a density-based approach. This tab will stay here until you refresh the app or pick a new dataset.

                        """)
        
        if 'filtered_df' in st.session_state:
            st.markdown("___")
            st.markdown(f"## Total records selected for clustering: **{len(st.session_state['filtered_df']):,}**")
            
            # Set final_clusters to widget defauls, unless user already selected one then keep the same
            if "final_clusters" not in st.session_state:
                st.session_state["final_clusters"] = 0

            if 'embeddings_df' not in st.session_state:
                st.session_state['embeddings_df'] = []
                
            st.session_state['embeddings_df'] = join_embeddings(st.session_state['filtered_df'], st.session_state['embeddings'])

            # User input for min and max clusters
            min_clusters, max_clusters = st.slider('Choose a range of clusters to explore', 2, 50, (10, 25))
            
            # Check and update the state for cluster range change
            cluster_range_changed = (min_clusters, max_clusters) != st.session_state.get('last_cluster_range', (None, None))
            if cluster_range_changed:
                st.session_state['last_cluster_range'] = (min_clusters, max_clusters)
                # Run K-Means Clustering related processes here

            if stateful_button.button("Run K-Means Clustering", key="run_kmeans"):
                st.markdown("`Please Note: You can unselect the button to re-start the analysis anytime`")
                # Check if clustering results are already calculated for the given range
                if 'kmeans_results_df' not in st.session_state or st.session_state['last_cluster_range'] != (min_clusters, max_clusters):
                    st.session_state['kmeans_results_df'] = fit_kmeans_and_get_metrics(st.session_state['embeddings_df'], EMBEDDING_COL_NAME, min_clusters, max_clusters)

                current_cluster_results = st.session_state['kmeans_results_df'].to_string(index=False)
                if "kmeans_topic_counts_str" not in st.session_state:
                    st.session_state.kmeans_topic_counts_str = current_cluster_results
                current_result_identifier = hash(current_cluster_results)
                kmeans_result_markdown = st.session_state['kmeans_results_df'].to_markdown(index=False)
                st.markdown("___")
                table_view, plot_view = st.tabs(["Results Table", "Results Plot"])

                with table_view:
                    st.markdown(kmeans_result_markdown)

                with plot_view:
                    # Check if plot is already generated
                    if 'results_plot' not in st.session_state or st.session_state['last_cluster_range'] != (min_clusters, max_clusters):
                        st.session_state['results_plot'] = plot_metrics(st.session_state['kmeans_results_df'])
                    st.plotly_chart(st.session_state['results_plot'], use_container_width=True)
                    st.session_state['last_cluster_range'] = (min_clusters, max_clusters)

                st.markdown("___")
                st.markdown("**Interpretation:**")
                
                # Check if feedback for the current result is already generated or needs updating
                if 'last_result_identifier' not in st.session_state or st.session_state['last_result_identifier'] != current_result_identifier:
                    with st.status("Getting GPT-4 interpretation...", expanded=True) as status:
                        messages = [
                            SystemMessage(content="You're an experienced data scientist specializing in document embedding clustering"),
                            HumanMessage(content=f"I have run a clustering analysis on OpenAI text embeddings and obtained the following results:\n\n{current_cluster_results}\n\nExplain the results in simple terms and suggest the best number of clusters for clear and distinct grouping."),
                        ]
                        cluster_feedback = gpt4.invoke(messages)
                        st.session_state['cluster_feedback'] = cluster_feedback.content
                        st.session_state['last_result_identifier'] = current_result_identifier
                        status.update(label="Analysis complete!", state="complete", expanded=True)

                st.markdown(st.session_state['cluster_feedback'])
                st.markdown("___")
                    
                # Prompt the user to enter a final n clusters
                final_clusters = st.selectbox('Select a number of clusters to:\n\n(1) Create a new "final" model\n\n(2) Generate topic labels with OpenAI', range(51), st.session_state['final_clusters'])
                # Check and update the state for final cluster count change
                final_cluster_count_changed = final_clusters != st.session_state.get('final_clusters', 0)
                if final_cluster_count_changed or st.button("Generate Clusters"):
                    st.session_state['final_clusters'] = final_clusters
                    
                    # Run final model and visualization related processes here         
                    with st.status("Generating clusters and getting labels from OpenAI...", expanded=True) as status:                 
                        if 'kmeans_bertopic_results_df' not in st.session_state or final_cluster_count_changed:
                            st.session_state['kmeans_bertopic_results_df'], st.session_state['kmeans_bertopic_model'], _ = fit_bertopic_and_get_labels_kmeans(st.session_state['embeddings_df'], final_clusters)
                        st.session_state["kmeans_topic_counts"] = st.session_state['kmeans_bertopic_results_df']['Top_n_words'].value_counts().reset_index().rename(columns={'index': 'Topic', 'Top_n_words': 'Count'}).sort_values('Count', ascending=False)
                        status.update(label=None, state="complete", expanded=True)
                
                    cluster_counts = st.session_state['kmeans_bertopic_results_df']['Top_n_words'].value_counts()
                    fig = go.Figure(data=[go.Bar(
                        y=[str(label) if len(str(label)) <= 50 else str(label)[:50] + "..." for label in cluster_counts.index],
                        x=cluster_counts.values, 
                        orientation='h'
                    )])
                    fig.update_layout(title='Number of Records per Cluster', xaxis_title='Number of Records', yaxis_title='Cluster Label')
                    st.session_state['cluster_count_fig'] = fig
                    
                    st.plotly_chart(st.session_state['cluster_count_fig'], use_container_width=True)
                            
                    with st.status("Creating scatter plot...", expanded=True) as status:
                        dims_2d = apply_tsne(st.session_state['kmeans_bertopic_results_df'], EMBEDDING_COL_NAME)
                        fig2 = plot_tsne_data(dims_2d, "Top_n_words")
                        st.session_state['scatter_fig'] = fig2
                        # Update so only re-runs if n_clusters is updated
                    st.session_state['final_clusters'] = final_clusters
                    status.update(label=None, state="complete", expanded=True)                                      

                    st.plotly_chart(st.session_state['scatter_fig'], use_container_width=False)
                        

        else:
            st.write("No filtered dataframe found. Please filter the dataframe in the 'Filter Data' tab.")

        # Main app functionality to guide users through a clustering analysis
    elif tab == "HDBSCAN Cluster Evaluation":
        st.markdown("___")
        with st.expander(label="Intro", expanded=False):
            st.markdown("""

    #### Workflow

    1. **Run Clustering:** Unlike `K-Means`, `HDBSCAN` does not require us to specify a number of clusters in advance. While `HDBSCAN` can be highly tuned and customized, here we just want the easy button.

    2. **Review Plots and Feedback:** Like before, you'll be presented with results along with feedback generated by GPT-4.

    3. **Visualize Results**

    4. **Iterative Analysis**
    
    5. **Proceed to the Next Tab**

                        """)
        
        if 'filtered_df' in st.session_state:
            st.markdown("___")
            st.markdown(f"## Total records selected for clustering: **{len(st.session_state['filtered_df']):,}**")

            # Check if embeddings are already calculated
            if 'embeddings_df' not in st.session_state:
                st.session_state['embeddings_df'] = []
                
            st.session_state['embeddings_df'] = join_embeddings(st.session_state['filtered_df'], st.session_state['embeddings'])

            if stateful_button.button("Run HDBSCAN Clustering", key="run_hdbscan"):
                st.markdown("`Please Note: You can unselect the button to re-start the analysis anytime`")
                with st.status("Generating clusters...", expanded=True) as status:
                    # Check if clustering results are already calculated for the given range
                    if 'hdbscan_results_df' not in st.session_state:
                        st.session_state['hdbscan_results_df'], cluster_info, topic_model_hdbscan = fit_bertopic_and_get_labels_hdbscan(st.session_state['embeddings_df'])
                    if 'cluster_info' not in st.session_state:
                        st.session_state['cluster_info'] = cluster_info    
                    if 'topic_model_hdbscan' not in st.session_state:
                        st.session_state['topic_model_hdbscan'] = topic_model_hdbscan

                    hdbscan_result_str = st.session_state['cluster_info'].to_string(index=False)
                    if "hdbscan_result_str" not in st.session_state:
                        st.session_state.hdbscan_result_str = hdbscan_result_str
                    # Generate a unique identifier for the current clustering result
                    hdbscan_current_result_identifier = hash(hdbscan_result_str)
                    hdbscan_result_markdown = st.session_state['cluster_info'].to_markdown(index=False)
                    status.update(label=None, state="complete", expanded=True)
                    
                st.markdown("___")
                st.markdown(f"Clustering algorithm found {len(np.unique(st.session_state['hdbscan_results_df']['Topic']))} topics")
                table_view, plot_view = st.tabs(["Table", "Plot"])
                with table_view:
                    st.markdown(hdbscan_result_markdown)

                with plot_view:
                    temp_df = st.session_state['cluster_info']
                    temp_df = temp_df[temp_df["Topic_Label"] != "Outlier"]
                    temp_df.sort_values(by="Count", ascending=False, inplace=True)
                    fig_hdbscan_counts = go.Figure(
                        data=[go.Bar(
                            y=temp_df.Topic_Label,
                            x=temp_df.Count, 
                            orientation='h'
                        )])
                    fig_hdbscan_counts.update_layout(
                        title='Number of Records per Cluster | Outliers Excluded', 
                        xaxis_title='Number of Records', 
                        yaxis_title='Cluster Label',
                        autosize=False,
                        width=1200,
                        height=800,
                        )
                    st.session_state['fig_hdbscan_counts'] = fig_hdbscan_counts
                    st.plotly_chart(fig_hdbscan_counts)

                st.markdown("___")
                st.markdown("**Interpretation:**")
                
                # Check if feedback for the current result is already generated
                if 'hdbscan_cluster_feedback' not in st.session_state or st.session_state['hdbscan_last_result_identifier'] != hdbscan_current_result_identifier:
                    with st.status("Getting GPT-4 interpretation...", expanded=True) as status:
                        messages = [
                            SystemMessage(content="You're an experienced data scientist specializing in document embedding clustering"),
                            HumanMessage(content=f"I have run a clustering analysis on OpenAI text embeddings using UMAP+HDBSCAN and obtained the following results:\n\n{hdbscan_result_str}\n\nPlease conduct a thorough analysis of the results including an overall assesment of the clustering. End with a markdown table summarizing the highlights from your observations."),
                        ]
                        cluster_feedback = gpt4.invoke(messages)
                        st.session_state['hdbscan_cluster_feedback'] = cluster_feedback.content
                        st.session_state['hdbscan_last_result_identifier'] = hdbscan_current_result_identifier
                        status.update(label=None, state="complete", expanded=True)

                st.markdown(st.session_state['hdbscan_cluster_feedback'])
                st.markdown("___")              
                
                temp_df = apply_umap(st.session_state['hdbscan_results_df'], EMBEDDING_COL_NAME)
                temp_fig = plot_umap_data(temp_df, "Topic_Label", max_line_length=50)
                st.plotly_chart(temp_fig)
                

        else:
            st.write("No filtered dataframe found. Please filter the dataframe in the 'Filter Data' tab.")
    
    
    elif tab == "Clustering Results":
        st.markdown("___")
        with st.expander(label="Intro", expanded=False):
            st.markdown("""

    #### Workflow

    1. **Run Report**
    """)
        st.markdown("___")
        if stateful_button.button("Run Clustering Report", key="run_final_report"):
            with st.status("Generating report...", expanded=False) as status:
        
        
                messages = [
                    SystemMessage(content="You're an experienced project manager overseeing a data science project on document embedding clustering"),
                    HumanMessage(content=f"I have run a comparative clustering analysis on OpenAI text embeddings using K-Means and UMAP+HDBSCAN. Please review the summaries from each clustering and provide feedback. Compare and contrast the results and summarize key takeaways. KMEANS_RESULTS:\n\n{st.session_state.kmeans_topic_counts_str}\n\nHDBSCAN_RESULTS:\n\n{st.session_state.hdbscan_result_str}\n\nConduct a thorough analysis of the results and end with a markdown table summarizing the highlights from your observations."),
                ]

                final_feedback = gpt4.invoke(messages)
                st.session_state['final_feedback'] = final_feedback.content
                # st.session_state['final_feedback_last_result_identifier'] = hdbscan_current_result_identifier
                status.update(label=None, state="complete", expanded=True)

                st.markdown(st.session_state['final_feedback'])
                st.markdown("___")

                csv = convert_df(st.session_state['hdbscan_results_df'])
                st.download_button(
                    label="Download data as CSV",
                    data=csv,
                    file_name='guided_clustering_results.csv',
                    mime='text/csv',
                )

main()

