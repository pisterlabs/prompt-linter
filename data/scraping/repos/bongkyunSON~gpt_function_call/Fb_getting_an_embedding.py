import openai
import os
from dotenv import load_dotenv
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-ada-002"


def get_quote_embedding(quote):
    response = openai.Embedding.create(
        model=EMBEDDING_MODEL,
        input=quote,
    )
    return response

print(get_quote_embedding("저녁 메뉴 추천해줘"))