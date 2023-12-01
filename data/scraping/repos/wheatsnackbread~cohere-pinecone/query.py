import os
import cohere
from dotenv import load_dotenv
import text_extractors
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
import pinecone


def query_pinecone(query):
    ##### Import environment variables
    load_dotenv()
    COHERE_API_KEY = os.getenv("COHERE_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

    ##### Define query
    # query = "Name the five dimensions of power."

    ##### Initialize Cohere and Pinecone
    co = cohere.Client(COHERE_API_KEY)

    pinecone.init(api_key=PINECONE_API_KEY, environment="gcp-starter")
    index_name = "lamsu2"
    # connect to index
    index = pinecone.Index(index_name)

    ##### Create the query embedding
    xq = co.embed(texts=[query], model="small", truncate="LEFT").embeddings

    # print(np.array(xq).shape)

    ##### Query, returning the top 3 most similar results
    res = index.query(xq, top_k=3, include_metadata=True)

    # for match in res["matches"]:
    # print(f"{match['score']:.2f}: {match['metadata']['text']}")

    return res
