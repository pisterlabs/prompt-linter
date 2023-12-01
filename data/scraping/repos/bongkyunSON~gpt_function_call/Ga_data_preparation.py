import openai
import pandas as pd
import os
from dotenv import load_dotenv
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

EMBEDDING_MODEL = "text-embedding-ada-002"
INPUT_DB_NAME = "Gx_reviews_database.csv"
OUTPUT_DB_NAME = "Gx_review_embeddings.csv"

df = pd.read_csv(INPUT_DB_NAME, usecols=["Summary", "Text", "Score"], nrows=500)
df = df[df["Score"] != 3] # 중립 점수인 3점을 택한 리뷰는 지움
df["Summ_and_Text"] = "Title: " + df["Summary"] + "; Content: " + df["Text"]


total_token_usage = 0
embeddings_generated = 0
total_data_rows = df.shape[0]

def get_embedding(item):
    global total_token_usage, embeddings_generated
    response = openai.Embedding.create(
        model=EMBEDDING_MODEL,
        input=item,
    )
    tokens_used = response["usage"]["total_tokens"]
    total_token_usage += tokens_used
    embeddings_generated += 1
    if (embeddings_generated % 10) == 0:
        print(
            f"지금까지 총 {total_token_usage}개의 토큰을 사용하여 {embeddings_generated}개의 임베딩을 생성했습니다. ({int((embeddings_generated / total_data_rows) * 100)}%)"
            
        )
    return response['data'][0]['embedding']

df["Embedding"] = df.Summ_and_Text.apply(lambda item: get_embedding(item))

df.to_csv(OUTPUT_DB_NAME, index=False)

print(
    f"""
총 {total_token_usage}개의 토큰이 사용된 {embeddings_generated}개의 임베딩을 생성했습니다. (완료!)
임베딩을 {OUTPUT_DB_NAME}에 저장했습니다.
"""
)

print(df.head(10))