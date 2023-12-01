from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings

load_dotenv()

text = "こんにちは。私はジョンです。よろしく。"

embeddings = OpenAIEmbeddings()
vector = embeddings.embed_query(text)
print(vector)
