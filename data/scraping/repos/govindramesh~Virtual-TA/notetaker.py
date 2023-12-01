from typing import Any

import uvicorn
from fastapi import FastAPI, Body
from pydantic import BaseModel

from langchain.llms import OpenAI
from langchain.embeddings import HuggingFaceEmbeddings
import pinecone
from langchain.embeddings import HuggingFaceEmbeddings


app = FastAPI()

embed = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")

YOUR_API_KEY = "278cc733-9151-48dc-8a00-21f445f8f38f"
YOUR_ENV = "gcp-starter"

index_name = 'virtual-ta'
pinecone.init(
    api_key=YOUR_API_KEY,
    environment=YOUR_ENV
)
index = pinecone.GRPCIndex(index_name)

llm = OpenAI(openai_api_key="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")

app.id = 1


# request input format
class Query(BaseModel):
    text: str


@app.get("/notes")
async def notes(
    query: Query = Body(...),
):
    prompt = f"""Write notes using bullet points or headers when appropriate, using the text below: \n
{query.text}
"""
    response = llm(prompt)
    return response

@app.get("/vectorize")
async def vectorize(
    query: Query = Body(...),
):
    metadata = {'text': query.text}
    embedding = embed.embed_query(query.text)
    id = f'vector-{app.id}'
    index.upsert([(id, embedding, metadata)])
    app.id += 1
    return app.id - 1

@app.get("/health")
async def health():
    """Check the api is running"""
    return {"status": "🤙"}
    

if __name__ == "__main__":
    uvicorn.run(
        "notetaker:app",
        host="localhost",
        port=8080,
        reload=True
    )
    print("hnotetakeri")
