# https://python.langchain.com/en/latest/modules/indexes/vectorstores/examples/pinecone.html
#https://blog.bytebytego.com/p/how-to-build-a-smart-chatbot-in-10


import getpass
from langchain import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA
import openai
import os
import pinecone
import requests
import json


OPENAI_API_KEY=os.getenv("OPEN_API_KEY")
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"),
              environment=os.getenv("PINECONE_ENVIRONMENT_KEY"))


embeddings = OpenAIEmbeddings()
index_name = "llm-demo"
docsearch = Pinecone.from_existing_index(index_name, embeddings)

#query = "What did the president say about Ketanji Brown Jackson"
query = "what types of abuses do you see"
docs = docsearch.similarity_search(query)


# print(docs[0].page_content)

def get_metadata(url):
  """Gets the metadata for a doc from Pinecone.

  Args:
    url: The URL of the doc.

  Returns:
    A dictionary containing the metadata for the doc.
  """

  # Make a request to Pinecone to get the metadata for the doc.
  response = requests.get(url)

  # If the request is successful, parse the response body to get the metadata.
  if response.status_code == 200:
    metadata = json.loads(response.content)
  else:
    raise Exception("Error getting metadata from Pinecone")

  # Return the metadata.
  return metadata

docs = docsearch.similarity_search(query)

def print_docs_with_metadata(docs):
  for doc in docs:
    print("Doc title:", doc["title"])
    print("Doc url:", doc["url"])
    print("Doc score:", doc["score"])
  #   doc["metadata"] = get_metadata(doc["url"])
  # print("Doc metadata:", doc["metadata"])

docs = docsearch.similarity_search(query)
print_docs_with_metadata(docs)

#  print(docs[0].page_content)
#  print(docs[0].url)