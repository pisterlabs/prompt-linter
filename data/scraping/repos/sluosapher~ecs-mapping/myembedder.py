import os
import openai
from typing import List

openai.api_key = os.environ.get('OPENAI_API_KEY')

DIMENSION_OF_EMBEDDINGS = 1536
EMBEDDING_MODEL = "text-embedding-ada-002"
EMBEDDINGS_BATCH_SIZE = 256 # The number of embeddings to request at a time
SIMILARITY_THRESHOLD = 0.76

# define a openai embedder class, inherit from Embedder
class MyEmbedder:
    def __init__(self):
        self.embedding_dim = DIMENSION_OF_EMBEDDINGS


    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Embed texts using OpenAI's ada model.
        Args:
            texts: The list of texts to embed.
        Returns:
            A list of embeddings, each of which is a list of floats.
        Raises:
            Exception: If the OpenAI API call fails.
        """
        # logging.debug(f"Getting embeddings for {len(texts)} texts")
        # Call the OpenAI API to get the embeddings
        response = openai.Embedding.create(input=texts, model=EMBEDDING_MODEL)
        # logging.debug(f"OpenAI response: {response}")

        # Extract the embedding data from the response
        data = response["data"]  # type: ignore

        # Return the embeddings as a list of lists of floats
        return [result["embedding"] for result in data]

    def get_embeddings_batch(self, chunk_texts: List[str]) -> List[List[float]]:
        """
        Embed texts in batch mode.
        Args:
            texts: The list of texts to embed.
        Returns:
            A list of embeddings, each of which is a list of floats.
        Raises:
            Exception: If the OpenAI API call fails.
        """
        embeddings: List[List[float]] = []
        for i in range(0, len(chunk_texts), EMBEDDINGS_BATCH_SIZE):
            # Get the text of the chunks in the current batch
            batch_texts = chunk_texts[i : i + EMBEDDINGS_BATCH_SIZE]

            # Get the embeddings for the batch texts
            # embedder = OpenAIEmbedder()
            # batch_embeddings = embedder.get_embeddings(batch_texts)

            response = openai.Embedding.create(input=batch_texts, model=EMBEDDING_MODEL)

            # Extract the embedding data from the response
            data = response["data"]  # type: ignore

            # Return the embeddings as a list of lists of floats
            batch_embeddings = [result["embedding"] for result in data]


            # Append the batch embeddings to the embeddings list
            embeddings.extend(batch_embeddings)

        return embeddings
