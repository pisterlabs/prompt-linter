import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

# Retrieves OpenAI embeddings for a given text
def get_embeddings(text):
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response['data'][0]['embedding']