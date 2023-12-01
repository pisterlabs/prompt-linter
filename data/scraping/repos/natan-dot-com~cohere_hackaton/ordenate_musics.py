import numpy as np
import cohere
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from python_tsp.exact import solve_tsp_dynamic_programming

from get_lyrics import get_lyrics
from get_lyrics_embedding import get_lyrics_embeddings
from get_k_songs_closes_to_centroid import get_k_songs_closes_to_centroid
from find_best_cluster import find_best_cluster

import typing as t



def ordenate_music(
    music_idxs: np.ndarray,
    embeddings: np.ndarray
) -> t.List[int]:
    target_embeddings = embeddings[music_idxs]
    distance_matrix = pairwise_distances(
        target_embeddings,
        target_embeddings,
        metric='cosine'
    )

    permutation, _ = solve_tsp_dynamic_programming(distance_matrix)
    return music_idxs[permutation].tolist()
