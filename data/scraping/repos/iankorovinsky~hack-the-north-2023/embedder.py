import cohere
from dotenv import dotenv_values

class StringEmbedder:
    def __init__(self, api_key):
        self.api_key = api_key

    def embed(self, text):
        client = cohere.Client(api_key=self.api_key)
        response = client.embed(texts=[text])
        embedding = response['embeddings'][0]
        return embedding

    @staticmethod
    def cosine_similarity(vector1, vector2):
        dot_product = sum(v1 * v2 for v1, v2 in zip(vector1, vector2))
        magnitude1 = sum(v1 ** 2 for v1 in vector1) ** 0.5
        magnitude2 = sum(v2 ** 2 for v2 in vector2) ** 0.5
        return dot_product / (magnitude1 * magnitude2)

# Example usage
env_config = dotenv_values(".env")
api_key = env_config.get('COHERE_API_KEY')
embedder = StringEmbedder(api_key)

text1 = "Hello, how are you?"
text2 = "Hi there!"

embedding1 = embedder.embed(text1)
embedding2 = embedder.embed(text2)

similarity = embedder.cosine_similarity(embedding1, embedding2)
print(f"Cosine similarity: {similarity}")  