import openai
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline
from yellowbrick.cluster import KElbowVisualizer
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

import process4positions as p4p


def basic_clustering(df, position, age_lower_threshold=16, age_upper_threshold=43, minute_threshold=540, n_components=2, ):

    df = df[df['Age'] >= f'{str(age_lower_threshold)}-000'][df['Age'] <= f'{str(age_upper_threshold)}-000']
    df = df[df['Minutes'] >= minute_threshold]
    df = df[df['DetailedPosition'].isin(p4p.position_name_dict[position])]
    # dataframe before processing and limiting amount of columns just to the ones used during clustering
    raw_df = deepcopy(df)
    if position !='Goalkeeper':
        p4p.calculate_new_metrics(raw_df)

    df = p4p.process(df, position)
    df.fillna(0, inplace=True)

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)

    # REASONABLE TO USE PIPELINE
    pca = PCA()
    tsne = TSNE(n_components=n_components, random_state = 42)
    pipeline = Pipeline([('pca', pca), ('tsne', tsne)])
    reduced_data = pipeline.fit_transform(scaled_data)

    #n_clusters = p4p.play_patterns_dict[position]
    visualizer = KElbowVisualizer(KMeans(random_state=42, n_init=10), k=(2,12), metric='distortion')
    visualizer.fit(reduced_data)
    n_clusters = visualizer.elbow_value_

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(reduced_data)

    # Join clusters and the df
    cluster_labels = pd.Series(cluster_labels)
    cluster_labels.index = raw_df.index
    clustered_df = pd.concat([cluster_labels, raw_df], axis=1)
    clustered_df.columns = ['Cluster']+list(raw_df.columns)
    clustered_df['Cluster'] = clustered_df['Cluster']+1

    return clustered_df


def beeswarm_comparison(clustered_df, metric, cluster2highlight):
    fig, ax = plt.subplots(figsize=(10, 2.5))
    plot_df = clustered_df[['Cluster', metric]]
    if plot_df[metric].dtype == 'object':
        plot_df[metric] = plot_df[metric].apply(lambda x: float(str(x).replace('%', '')))

    palette = dict(zip(clustered_df['Cluster'].unique(), ['#fafafa']*len(clustered_df['Cluster'].unique())))
    palette[cluster2highlight] = '#ff4b4b'

    sns.swarmplot(data=plot_df, x=metric, hue='Cluster', palette=palette, ax=ax, legend=None)
    fig.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#fafafa')
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    ax.tick_params(axis='x', colors='#fafafa')
    plt.suptitle(f'{metric}', color='#fafafa', size=16, fontweight="bold", family="monospace", ha='center')
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)

    return fig


def chatgpt_call(prompt, model="gpt-3.5-turbo"):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0, 
    )
    return response.choices[0].message["content"]


def generate_AI_analysis(df_cluster, position):
    cluster_description = df_cluster.describe()

    prompt = f"""
    You are a football analyst. Your task is to describe the player profile for a cluster of players. \
    Their position is {position}. \
    You will be given a dataframe summarizing the metrics used for analysis within our cluster.\
    If not stated differently, the stats are per 90 minutes. In Your analysis take into consideration AS MANY stats as You can. \
    YOU MUSTN'T MENTION THE NUMBERS DIRECTLY, focus on the style of play that summary numbers indicate. \
    Make sure Your analysis ends with paragraph describing THE STYLE of play for our cluster. \
    The dataframe will delimited by 3 backticks, i.e. ```.

    ```{cluster_description}```
    
    """

    return chatgpt_call(prompt)