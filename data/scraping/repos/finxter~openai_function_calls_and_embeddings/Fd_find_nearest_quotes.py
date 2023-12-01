import openai
import numpy as np
import pandas as pd

from decouple import config
from openai.embeddings_utils import cosine_similarity

openai.api_key = config("CHATGPT_API_KEY")
EMBEDDING_MODEL = "text-embedding-ada-002"


def get_quote_embedding(quote):
    response = openai.Embedding.create(
        model=EMBEDDING_MODEL,
        input=quote,
    )
    return response["data"][0]["embedding"]


df = pd.read_csv("Fx_embedding_db.csv")
df["embedding"] = df.embedding.apply(eval).apply(np.array)


def find_similar_quotes(user_input, number_of_results=5):
    user_embedding = get_quote_embedding(user_input)
    df["similarity"] = df.embedding.apply(
        lambda embedding: cosine_similarity(embedding, user_embedding)
    )
    result = df.sort_values(by="similarity", ascending=False).head(number_of_results)
    for i in range(number_of_results):
        print(
            f"{i+1}: {result.iloc[i]['quote']} - {result.iloc[i]['author']} ({result.iloc[i]['similarity']})"
        )
    return result


try:
    while True:
        print(
            "Welcome to the quote finder! Please enter a quote to find similar quotes."
        )
        user_quote = input("Enter a quote or press ctrl+c to exit: ")
        result = find_similar_quotes(user_quote)
except KeyboardInterrupt:
    print("Exiting...")
