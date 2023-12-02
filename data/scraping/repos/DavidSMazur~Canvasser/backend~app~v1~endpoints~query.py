# /v1/endpoints/query.py
from pydantic import BaseModel

from pymongo import MongoClient
from pymongo.server_api import ServerApi
import certifi
import os
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain.embeddings.openai import OpenAIEmbeddings
import json
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
import re


from fastapi import APIRouter
import dotenv

from openai import OpenAI
ai_client = OpenAI()

INSTRUCTION = "I just gave you a lot of data about assignments and announcements for the course I'm currently taking. I also provided a query that I want you to answer. Make sure you answer the query with correct punctuation and grammar. Try to be as friendly as possible."

dotenv.load_dotenv()
ca = certifi.where()

router = APIRouter()


# Define the request model
class QueryRequest(BaseModel):
    course_id: str
    query_text: str


def get_db():
    uri = os.getenv('MONGO_COLLECTION_STRING')
    client = MongoClient(uri, server_api=ServerApi('1'), tlsCAFile=ca)
    db = client.MHacks
    return db


@router.post("/")
async def query(request: QueryRequest):
    embeddings = OpenAIEmbeddings()


    db = get_db()
    client = db.client  # Get the MongoClient object
    collection_name = "embeddings"

    collection = db[collection_name]
    index_name = "default"



    query = request.query_text
    # print(embeddings.embed_query(query))

    results = collection.aggregate([
      {"$vectorSearch": {
        "queryVector": embeddings.embed_query(query),
        "path": "embedding",
        "numCandidates": 100,
        "limit": 4,
        "index": "default",
          }}
    ])

    # Convert the CommandCursor to a list and then to a string
    results_list = list(results)
    print(results_list[0].get("text"))


    # Extract top 3 results
    top_results = results_list[:1]

    # Format results for use in OpenAI API (you might need to summarize or reformat this data)
    formatted_results = [result.get("text") for result in top_results]
    print(formatted_results)

    formatted_results_no_brackets = ""

    for res in formatted_results:
        formatted_results_no_brackets += res.replace("{", "").replace("}", "").replace('"', "").replace("text: ", "") + " "

    # print(formatted_results_no_brackets)

    prompt = PromptTemplate.from_template(
        "I am implementing a RAG model. As a result, I will give you the top 1 results from a similarity search and I will also provide you the query, Your job is to condense the top 1 results into 1 response that best answers the prompt. Here are the top 1 results formatted_results: " + str(formatted_results_no_brackets) + " Here is the query: {query} Remove any json tags like curly brackets, quotation marks or ID labels. The response should be a paragraph, made up of full sentences. Respond as if only the query was asked directly to you."
    )


    runnable = prompt | ChatOpenAI() | StrOutputParser()
    response = runnable.invoke({"query": query})

    # print(response)
    
    # results_str = str(results_list)

    # print(json.dumps(str(results_list)))
    # for document in results:
    #     print(document.text)

    return {"response": response}

    # return {"response": response.choices[0].message}
    # return {"response": results_list[0].get("text")}


# @router.post("/", response_model=ItemSchema)
# async def create_item(item: ItemSchema):
#     return item
