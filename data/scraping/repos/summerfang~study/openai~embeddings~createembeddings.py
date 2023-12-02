import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

embedding = openai.Embedding.create(
    model="text-embedding-ada-002",
    input = "The food is delicious and the waiter ..."
)

print(embedding)