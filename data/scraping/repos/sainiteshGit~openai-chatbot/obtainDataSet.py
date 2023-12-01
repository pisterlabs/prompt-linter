# imports
import pandas as pd
import tiktoken
import openai
from openai.embeddings_utils import get_embedding
openai.api_key = 'YOUR_OPEN_API_KEY'

####################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################### embedding model parameters ######################################################################################################
embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
max_tokens = 8000  # the maximum for text-embedding-ada-002 is 8191


# load & inspect dataset
input_datapath = "/Users/SaiNitesh/Projects/openai-chatbot/content/data/search/Reviews.csv"  # to save space, we provide a pre-filtered dataset
df = pd.read_csv(input_datapath, index_col=0)
df = df[["Time", "ProductId", "UserId", "Score", "Summary", "Text"]]
df = df.dropna()
df["combined"] = (
    "Title: " + df.Summary.str.strip() + "; Content: " + df.Text.str.strip()
)
df.head(2)

##### subsample to 1k most recent reviews and remove samples that are too long
top_n = 500
df = df.sort_values("Time").tail(top_n * 2)  # first cut to first 2k entries, assuming less than half will be filtered out
df.drop("Time", axis=1, inplace=True)

encoding = tiktoken.get_encoding(embedding_encoding)

# omit reviews that are too long to embed
df["n_tokens"] = df.combined.apply(lambda x: len(encoding.encode(x)))


df = df[df.n_tokens <= max_tokens].tail(top_n)

len(df)

df["embedding"] = df.combined.apply(lambda x: get_embedding(x, engine=embedding_model))
df.to_csv("/Users/SaiNitesh/Projects/openai-chatbot/content/data/search/fine_food_reviews_with_embeddings_500.csv")