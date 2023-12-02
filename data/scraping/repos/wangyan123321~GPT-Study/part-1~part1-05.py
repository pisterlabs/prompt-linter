import pandas as pd
import tiktoken
import openai
import os

from openai.embeddings_utils import get_embedding, get_embeddings

openai.api_key = os.environ.get("OPENAI_API_KEY")

embedding_model = "text-embeddings-ada-002"
embedding_encoding = "cl100k_base"
max_tokens = 8000

df = pd.read_csv('toutiao_cat_data.txt', sep='_!_', names=['id', 'code', 'category', 'title', 'keywords'])
df = df.fillna("")
df["combined"] = (
        "标题：" + df.title.str.strip() + ": 关键字：" + df.keywords.str.strip() + ": 分类：" + df.category.str.strip()
)

for i in range(1, 10):
    print(df.combined[i] + "\n")

encoding = tiktoken.get_encoding(embedding_encoding)

df["n_tokens"] = df.combined.apply(lambda x: len(encoding.encode(x)))
df = df[df.n_tokens <= max_tokens]

print(df.n_tokens.describe())
