import openai
from decouple import config

openai.api_key = config("CHATGPT_API_KEY")
EMBEDDING_MODEL = "text-embedding-ada-002"


def get_quote_embedding(quote):
    response = openai.Embedding.create(
        model=EMBEDDING_MODEL,
        input=quote,
    )
    return response


print(get_quote_embedding("Please embed this sentence for me!"))
