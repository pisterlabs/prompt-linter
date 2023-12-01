# ObtainDataEmbedding.py
# imports
import openai
import pandas as pd

import tiktoken
from openai.embeddings_utils import get_embedding
import config
# set your API key
openai.api_key = config.OPENAI_API_KEY

# embedding model parameters
embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
max_tokens = 8000  # the maximum for text-embedding-ada-002 is 8191

# load & inspect dataset
input_datapath = "data/chatbotdata.csv"
df = pd.read_csv(input_datapath, index_col=0)
df = df[["userid", "chathistory", "avoide", "avoida", "avoidb", "avoidc", "avoidd", "anxietye", "anxietya", "anxietyb", "anxietyc", "anxietyd"]]
df = df.dropna()
df.head(2)

# Filter out chat transcripts that are too long to embed, estimate for the maximum number of words would be around 1638 words (8191 tokens / 5).
encoding = tiktoken.get_encoding(embedding_encoding)

df["n_tokens"] = df.chathistory.apply(lambda x: len(encoding.encode(x)))
df = df[df.n_tokens <= max_tokens]
len(df)

# Ensure you have your API key set in your environment per the README: https://github.com/openai/openai-python#usage

# This may take a few minutes
df["embedding"] = df.chathistory.apply(lambda x: get_embedding(x, engine=embedding_model))
df.to_csv("data/chat_transcripts_with_embeddings_and_scores3.csv")


# Please replace "data/chat_transcripts.csv" with the path to your actual data file. Also, replace 'ChatTranscript', 'Attachment', 'Avoidance' with the actual column names of your chat transcripts and attachment scores in your data file.

# Also, remember to set the API key for OpenAI in your environment before running the get_embedding function.