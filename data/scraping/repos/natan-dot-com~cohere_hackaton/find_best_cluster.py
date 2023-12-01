import numpy as np
import cohere
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from get_lyrics import get_lyrics
from get_lyrics_embedding import get_lyrics_embeddings
from get_k_songs_closes_to_centroid import get_k_songs_closes_to_centroid


def find_best_cluster(
    co: cohere.Client,
    kmeans: KMeans,
    prompt: str,
    model = 'embed-english-v2.0',
    metric = 'cosine',
) -> int:
    """
        Returns the index of the best cluster
    """
    assert len(prompt) > 0, 'Cannot find best cluster from empty prompt'
    response = co.embed(
        texts=[prompt],
        model=model,
        truncate='START'
    )
    prompt_embed = np.array(response.embeddings[0])
    prompt_embed = prompt_embed.reshape(1,-1)
    centroids = kmeans.cluster_centers_
    distances = pairwise_distances(prompt_embed, centroids, metric=metric)
    closest_idx = np.argmin(distances)

    return closest_idx
