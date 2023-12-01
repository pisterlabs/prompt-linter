import openai
import pandas as pd
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from dotenv import load_dotenv
load_dotenv()
from Fx_quotes import quotes

openai.api_key = os.getenv("OPENAI_API_KEY")

EMBEDDING_MODEL = "text-embedding-ada-002"

total_tokens_used = 0
total_embeddings = 0

def get_quote_embedding(quote):
    global total_tokens_used, total_embeddings
    response = openai.Embedding.create(
        model=EMBEDDING_MODEL,
        input=quote,
    )
    tokens_used = response["usage"]["total_tokens"]
    total_tokens_used += tokens_used
    total_embeddings += 1
    if (total_embeddings % 10) == 0:
        print(
            f"생성된 임배딩 수: {total_embeddings} // 현재까지 사용된 토큰 수: {total_tokens_used}. ({int((total_embeddings / len(quotes)) * 100)}%)"
        )
    return response["data"][0]["embedding"]


embedding_df = pd.DataFrame(columns=["quote", "author", "embedding"])

for index, quote in enumerate(quotes):
    current_quote = quote[0]
    try:
        current_author = quote[1]
    except IndexError:
        current_author = "Unknown"
    embedding = get_quote_embedding(current_quote)
    embedding_df.loc[index] = [current_quote, current_author, embedding]


embedding_df.to_csv("Fx_embedding_db.csv", index=False)

print(
    f"""
총 {total_tokens_used}개의 토큰이 사용된 {total_embeddings}개의 임베딩을 생성했습니다. 
임베딩을 Embedding_db.csv에 저장하고 데이터 프레임 헤드를 인쇄했습니다: {embedding_df.head(5)}
"""
)