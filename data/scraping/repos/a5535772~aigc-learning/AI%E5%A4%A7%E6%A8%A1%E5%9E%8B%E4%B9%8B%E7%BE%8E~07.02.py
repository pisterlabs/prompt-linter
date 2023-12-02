import backoff
import openai
import os
import pandas as pd
import tiktoken
from openai.embeddings_utils import get_embeddings

openai.api_key = os.environ.get("OPENAI_API_KEY")
embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
batch_size = 2000
max_tokens = 8000  # the maximum for text-embedding-ada-002 is 8191

df = pd.read_csv('data/20_newsgroup.csv')
print("Number of rows before null filtering:", len(df))
df = df[df['text'].isnull() == False]
encoding = tiktoken.get_encoding(embedding_encoding)

df["n_tokens"] = df.text.apply(lambda x: len(encoding.encode(x)))
print("Number of rows before token number filtering:", len(df))
df = df[df.n_tokens <= max_tokens]
print("Number of rows data used:", len(df))


# 用lambda表达式打印df的每一行的text
# df.text.apply(lambda x: print(x))


@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def get_embeddings_with_backoff(prompts, engine):
    embeddings = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        embeddings += get_embeddings(list_of_text=batch, engine=engine)
    return embeddings


prompts = df.text.tolist()
prompt_batches = [prompts[i:i + batch_size] for i in range(0, len(prompts), batch_size)]

embeddings = []
for batch in prompt_batches:
    batch_embeddings = get_embeddings_with_backoff(prompts=batch, engine=embedding_model)
    embeddings += batch_embeddings

df["embedding"] = embeddings
df.to_parquet("data/20_newsgroup_with_embedding.parquet", index=False)
