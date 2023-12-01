from fastapi import FastAPI
from pydantic import BaseModel

from ast import literal_eval


import os
import langchain
from langchain.vectorstores import FAISS

from langchain.embeddings.openai import OpenAIEmbeddings



app = FastAPI()


class Stock(BaseModel):
    ticker: str
    ve: str


@app.post("/related")
async def read_item(item: Stock ):


    embeddings = OpenAIEmbeddings(openai_api_key="sk-asafdoesntreallymatteryft652174u1y")


    # print(item)
    new_db = FAISS.load_local(str(item.ticker), embeddings)

    # var array = JSON.parse("[" + item.ve + "]")

    ar = literal_eval(item.ve)
    docs_and_scores = new_db.similarity_search_by_vector(ar)


    size = len(docs_and_scores)

    data = ""
    for i in range(0,size):
        data+=(docs_and_scores[i].page_content + "\n\n")

    return data  
