from langchain.embeddings.openai import OpenAIEmbeddings
from config import OPENAI_API_KEY

model_name = 'text-embedding-ada-002'
embed = OpenAIEmbeddings(
    model=model_name,
    openai_api_key=OPENAI_API_KEY
)

def getDocEmbeddings(content):
    embeddings = embed.embed_documents(content)
    return embeddings

def getQueryEmbeddings(query):
    query_result = embed.embed_query(query)
    return query_result