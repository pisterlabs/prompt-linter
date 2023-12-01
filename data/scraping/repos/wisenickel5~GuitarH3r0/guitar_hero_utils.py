from openai import AzureOpenAI
from openai.types import CreateEmbeddingResponse
import numpy as np
import re
from typing import List


def normalize_text(s: str, sep_token: str = " \n ") -> str:
    """
    Normalizes the input text by removing extra whitespace and correcting punctuation.

    This function is typically used to prepare text data for NLP tasks like tokenization
    or embedding generation.

    :parameter s: The text to normalize.
    :parameter sep_token: A separator token used for joining text segments, defaulted to a space followed by a newline.

    :return: The normalized text string.

    The function applies several regular expressions and string operations to clean up the text:
    - Collapses multiple whitespace characters into a single space.
    - Removes any occurrences of a space followed by a comma.
    - Replaces instances of double periods or spaced periods with a single period.
    - Strips leading and trailing whitespace characters.
    """

    # Collapse one or more whitespace characters into a single space and strip leading/trailing spaces.
    s = re.sub(r'\s+', ' ', s).strip()
    # Remove occurrences of a space followed by a comma, which is typically a punctuation error.
    s = re.sub(r"\s+,", ",", s)
    # Replace instances of double periods or spaced periods with a single period.
    s = s.replace("..", ".").replace(". .", ".")
    # Remove newline characters and any additional whitespace that may have been added.
    s = s.replace(sep_token, " ").strip()
    return s


def get_embedding(text: str, c: AzureOpenAI, model="Guitar-H3r0-Embeddings") -> List[float]:
    """
    Retrieve an embedding for the given text using Azure's OpenAI service.

    :parameter text: The input text string to be converted into an embedding.
    :parameter c: An instantiated AzureOpenAI client configured to communicate with the OpenAI service.
    :parameter model: The model identifier for the deployed embedding model to use.

    :return: A list of floating-point numbers representing the text embedding.

    The function normalizes the input text by removing redundant whitespace and
    unnecessary punctuation, then requests the OpenAI API to create an embedding
    for the processed text. It returns the embedding as a list of floats.
    """

    # Normalize the input text to remove redundant spaces and fix punctuation.
    text = normalize_text(text)

    # Call the OpenAI API to create an embedding for the normalized text.
    # 'model' should correspond to a valid deployment model identifier.
    response: CreateEmbeddingResponse = c.embeddings.create(input=[text], model=model)

    # Return the embedding from the API response.
    # The API returns a list of embeddings; we take the first one since we provided a single input.
    return response.data[0].embedding


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate the cosine similarity between two vectors.
    :parameter: vec1 (array): A numpy array representing the first vector.
    :parameter: vec2 (array): A numpy array representing the second vector.

    :return: float: cosine similarity between vec1 and vec2.
    """
    # Compute the dot product between the two vectors
    dot_product = np.dot(vec1, vec2)
    # Since OpenAI embeddings are normalized, the vectors' magnitudes are both 1,
    # so we don't need to divide by the product of magnitudes.
    return dot_product
