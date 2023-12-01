import openai
import pandas as pd

from decouple import config

from Fx_quotes import quotes

openai.api_key = config("CHATGPT_API_KEY")
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
            f"Generated {total_embeddings} embeddings so far with a total of {total_tokens_used} tokens used. ({int((total_embeddings / len(quotes)) * 100)}%)"
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
Generated {total_embeddings} embeddings with a total of {total_tokens_used} tokens used. (Done!)
Succesfully saved embeddings to embedding_db.csv, printing dataframe head:
{embedding_df.head(5)}

    """
)
