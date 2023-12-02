import openai
import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv
load_dotenv()
from openai.embeddings_utils import cosine_similarity

openai.api_key = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-ada-002"

def get_quote_embedding(quote):
    response = openai.Embedding.create(
        model=EMBEDDING_MODEL,
        input=quote,
    )
    return response["data"][0]["embedding"]

df = pd.read_csv("/Users/sbk/gpt_function/gpt_function_call/F_quote_embeddings/Fx_embedding_db.csv")
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
            "검색기에 오신 것을 환영합니다! 텍스트를 넣으시면 비슷한 인용문을 찾아드립니다."
        )
        user_quote = input("엔터를 눌러 입력하거나 끝내시려면 ctrl + c를 입력하세요: ")
        result = find_similar_quotes(user_quote)
except KeyboardInterrupt:
    print("종료중...")