
from openai.embeddings_utils import get_embeddings
import openai, os, tiktoken, backoff
import pandas as pd

openai.api_key = os.environ.get("OPENAI_API_KEY")
embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
batch_size = 2000
max_tokens = 8000  # the maximum for text-embedding-ada-002 is 8191

# 1. 对数据做预处理，过滤掉数据里面有些文本是空的情况，以及把 Token 数量太多的给过滤掉。
df = pd.read_csv('20_newsgroup.csv')
print("Number of rows before null filtering:", len(df))
df = df[df['text'].isnull() == False]
encoding = tiktoken.get_encoding(embedding_encoding)

df["n_tokens"] = df.text.apply(lambda x: len(encoding.encode(x)))
print("Number of rows before token number filtering:", len(df))
df = df[df.n_tokens <= max_tokens]
print("Number of rows data used:", len(df))

# 2. 通过 Embedding 的接口，拿到文本的 Embedding 向量，然后把整个数据存储成 parquet 文件

# 如果你直接一条条调用 OpenAI 的 API，很快就会遇到报错。这是因为 OpenAI 对 API 的调用进行了限速（Rate Limit）。如果你过于频繁地调用，就会遇到限速的报错。而如果你在报错之后继续持续调用，限速的时间还会被延长。那怎么解决这个问题呢？选用 backoff 这个 Python 库，在调用的时候如果遇到报错了，就等待一段时间，如果连续报错，就拉长等待时间。
# 通过 backoff 库，我们指定了在遇到 RateLimitError 的时候，按照指数级别增加等待时间。
@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def get_embeddings_with_backoff(prompts, engine):
    embeddings = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        embeddings += get_embeddings(list_of_text=batch, engine=engine)
    return embeddings

prompts = df.text.tolist()
# range(start, stop, step), range(0， 5) 等价于 range(0, 5, 1)
prompt_batches = [prompts[i:i+batch_size] for i in range(0, len(prompts), batch_size)]

embeddings = []
count = 1
for batch in prompt_batches:
    print('begin to get ' + str(count) + ' batch')
    count = count+1
    batch_embeddings = get_embeddings_with_backoff(prompts=batch, engine=embedding_model)
    embeddings += batch_embeddings

df["embedding"] = embeddings
df.to_parquet("20_newsgroup_with_embedding.parquet", index=False)