#!/usr/bin/env python
import os

import lancedb
from lancedb.embeddings import with_embeddings
import openai
import pandas as pd
from pydantic import BaseModel, Field
import requests

from aifunctools.openai_funcs import complete_with_functions

openai.api_key = os.getenv("OPENAI_API_KEY")

MODEL = "gpt-3.5-turbo-16k-0613"

db = lancedb.connect(".lancedb")


def embed_func(c):
    rs = openai.Embedding.create(input=c, engine="text-embedding-ada-002")
    return [record["embedding"] for record in rs["data"]]


def to_lancedb_table(db, memes):
    df = pd.DataFrame([m.model_dump() for m in memes])
    data = with_embeddings(embed_func, df, column="name")
    if "memes" in db.table_names():
        tbl = db.open_table("memes")
        tbl.add(data, mode="overwrite")
    else:
        tbl = db.create_table("memes", data)
    return tbl


class Meme(BaseModel):
    id: str = Field(description="The meme id")
    name: str = Field(description="The meme name")
    url: str = Field(description="The meme url")
    width: int = Field(description="The meme image width")
    height: int = Field(description="The meme image height")
    box_count: int = Field(description="The number of text boxes in the meme")


def get_memes():
    """
    Get a list of memes from the meme api
    """
    resp = requests.get("https://api.imgflip.com/get_memes")
    return [Meme(**m) for m in resp.json()["data"]["memes"]]


def search_memes(query: str):
    """
    Get the most popular memes from imgflip and do a semantic search based on the user query

    :param query: str, the search string
    """
    memes = get_memes()
    tbl = to_lancedb_table(db, memes)
    df = tbl.search(embed_func(query)[0]).limit(1).to_df()
    return Meme(**df.to_dict(orient="records")[0]).model_dump()


if __name__ == "__main__":
    question = "Please find me the image link for that popular meme with Fry from Futurama"
    print(complete_with_functions(question, search_memes)["choices"][0]["message"]["content"])
