from fastapi import FastAPI, HTTPException, Depends, Request
import os 
from dotenv import load_dotenv
# import constants
import os
# os.environ["OPENAI_API_KEY"] = constants.APIKEY
from langchain.vectorstores import FAISS, Chroma
from langchain.embeddings import OpenAIEmbeddings
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import chromadb
load_dotenv()

def load_vectorstore():
    embeddings = OpenAIEmbeddings()
    persistent_client = chromadb.PersistentClient("chroma_db")
    vectorstore = Chroma(client=persistent_client, embedding_function=embeddings, collection_name="products")
    return vectorstore

vectorstore = load_vectorstore()

app = FastAPI()


def veryify_api_key(request: Request):
    token = request.headers.get("Authorization")
    if not token: 
        raise HTTPException(status_code=401, detail="No Api Key provided in the header")
    token_str = token.split(" ")[1]
    if token_str == os.getenv("SHOPIFY_STATIC_TOKEN"):
        return token
    else:
        raise HTTPException(status_code=401, detail="Invalid API Key")

@app.get("/product_search")
async def product_search(query: str):
    try:
        results = vectorstore.similarity_search(query, k=2)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))