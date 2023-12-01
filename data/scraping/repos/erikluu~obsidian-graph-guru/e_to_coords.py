import sys
import os
import json
import openai
import numpy as np
from sklearn.decomposition import PCA
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

def reduce_dimensions(vec_embeddings):
    pca = PCA(n_components=2)
    pca.fit(vec_embeddings)
    pca_encodings = pca.transform(vec_embeddings)
    return pca_encodings

with open('../../results/embeddings.json','r') as infile:
        data = json.load(infile)

embedding_labels = list(data.keys())
# embedding_size = len(data[embedding_labels[0]])
embeddings = np.array([data[label] for label in embedding_labels])
latent_space = reduce_dimensions(embeddings)
coordinates = dict(zip(embedding_labels, latent_space.tolist()))

with open('../../results/coordinates.json', 'w') as fp:
        json.dump(coordinates, fp)
