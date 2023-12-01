# Copyright (c) 2023 Eliah Kagan
#
# Permission to use, copy, modify, and/or distribute this software for any
# purpose with or without fee is hereby granted.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
# REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY
# AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
# INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
# LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
# OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
# PERFORMANCE OF THIS SOFTWARE.

"""
Computing text embeddings of repository names.

The text embeddings computed are computed with OpenAI's new (in December 2022)
model text-embedding-ada-002.
"""

__all__ = ['EmbeddingsMatrix', 'embed', 'embed_many']

from nptyping import Float32, NDArray, Shape, assert_isinstance
import numpy as np
import openai.embeddings_utils

from fr2ex import _task

EmbeddingVector = NDArray[Shape['1536'], Float32]
"""An embedding in a 1536-dimensional space."""

EmbeddingsMatrix = NDArray[Shape['*, 1536'], Float32]
"""A matrix whose rows are embeddings in a 1536-dimensional space."""


def embed(text: str) -> EmbeddingVector:
    """Query the API for text-embedding-ada-002 for the text. No caching."""
    _task.ensure_api_key()
    embedding = openai.embeddings_utils.get_embedding(
        text=text,
        engine='text-embedding-ada-002',
    )
    column_vector = np.array(embedding, np.float32)
    assert_isinstance(column_vector, EmbeddingVector)
    return column_vector


@_task.api_task('embeddings')
def embed_many(texts: list[str]) -> EmbeddingsMatrix:
    """Load or query the API for text-embedding-ada-002 for all texts."""
    embeddings = openai.embeddings_utils.get_embeddings(
        list_of_text=texts,
        engine='text-embedding-ada-002',
    )
    matrix = np.array(embeddings, np.float32)
    assert_isinstance(matrix, EmbeddingsMatrix)
    return matrix
