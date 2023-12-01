"""
This script demonstrates the use of text embeddings to compare semantic similarity
between different sets of data. 

Text embeddings are a powerful technique in natural language processing (NLP) that
convert text into numerical vectors. These vectors represent the text in a
high-dimensional space, where similar meanings are close together, and different
meanings are further apart. This allows for the quantitative comparison of textual data
based on their semantic content rather than just syntactical similarity.

In this script, we use the OpenAI API to create embeddings for different sets of text
data - one set containing names of animals and another containing colors. We then
calculate the cosine similarity between these embedding vectors to determine how similar
the sets of data are to each other. The cosine similarity is a measure that calculates
the cosine of the angle between two vectors, giving us an indication of how similar the
meanings of the two text sets are.

The script is useful for understanding how embeddings can be applied in real-world
scenarios to compare and analyze text data based on their underlying semantics.
"""

from dotenv import load_dotenv
from openai import OpenAI
from scipy.spatial.distance import cosine

load_dotenv()

# Initialize the OpenAI client
client = OpenAI()


def compare_embeddings(embedding1: list[float], embedding2: list[float]) -> float:
    """Calculate the cosine similarity between two embedding vectors."""
    return 1 - cosine(embedding1, embedding2)


if __name__ == "__main__":
    # Define sample data strings
    data_animals1 = "Dogs, Cats, Mice, Birds, Fish, Elephants"
    data_animals2 = "Cats, Dogs, Birds, Fish, Elephants, Mice"
    data_colors = "Red, Green, Blue, Yellow, Orange, Purple"

    # Generate embeddings for each data string using the OpenAI Embeddings API
    response_animals1 = client.embeddings.create(
        input=data_animals1,
        model="text-embedding-ada-002",
    )
    response_animals2 = client.embeddings.create(
        input=data_animals2,
        model="text-embedding-ada-002",
    )
    response_colors = client.embeddings.create(
        input=data_colors,
        model="text-embedding-ada-002",
    )

    # Extract the embedding vectors from the API responses
    embedding_animals1 = response_animals1.data[0].embedding
    embedding_animals2 = response_animals2.data[0].embedding
    embedding_colors = response_colors.data[0].embedding

    # Compare the embeddings to evaluate similarity
    similarity_animals = compare_embeddings(embedding_animals1, embedding_animals2)
    similarity_animals_colors1 = compare_embeddings(
        embedding_animals1, embedding_colors
    )
    similarity_animals_colors2 = compare_embeddings(
        embedding_animals2, embedding_colors
    )

    # Print the similarity results
    print(f"Similarity between animals datasets: {similarity_animals}")
    print(f"Similarity between animals1 and colors: {similarity_animals_colors1}")
    print(f"Similarity between animals2 and colors: {similarity_animals_colors2}")
