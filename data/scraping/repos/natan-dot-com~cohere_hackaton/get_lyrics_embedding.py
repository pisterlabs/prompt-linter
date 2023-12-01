import typing as t
import logging
import cohere
from get_lyrics import get_lyrics
import numpy as np

logger = logging.getLogger(__name__)

def get_lyrics_embeddings(
    co: cohere.Client,
    texts: t.List[str],
    model = 'embed-english-v2.0',
) -> np.ndarray:
    logger.info("getting embeddings")
    assert len(texts) > 0, "lyrics list is empty"
    assert len(texts) <= 96, "Maximum number of texts per call is 96"
    response = co.embed(
        texts=texts,
        model=model,
        truncate='START'
    )

    embeddings = response.embeddings
    return np.array(embeddings)
