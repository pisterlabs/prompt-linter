# https://python.langchain.com/docs/integrations/vectorstores/mongodb_atlas

# issue - It is not returning any results

import os
import getpass

os.environ["OPENAI_API_KEY"] = "<YOUR_API_KEY>"
MONGODB_ATLAS_CLUSTER_URI = "<YOUR_URI>"

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain.document_loaders import TextLoader
from langchain.docstore.document import Document

from pymongo import MongoClient

loader = TextLoader("state_of_the_union.txt")

documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()


# initialize MongoDB python client
client = MongoClient(MONGODB_ATLAS_CLUSTER_URI)

db_name = "langchain_db"
collection_name = "langchain_col"
collection = client[db_name][collection_name]
index_name = "langchain_demo"

# insert the documents in MongoDB Atlas with their embedding
docsearch = MongoDBAtlasVectorSearch.from_documents(
    docs, embeddings, collection=collection, index_name=index_name
)

query = "What did the president say about Ketanji Brown Jackson"
# docs_with_score = docsearch.similarity_search_with_score(query)

# for doc, score in docs_with_score:
#     print("-" * 80)
#     print("Score: ", score)
#     print(doc.page_content)
#     print("-" * 80)

# perform a similarity search between the embedding of the query and the embeddings of the documents
query = "What did the president say about Ketanji Brown Jackson"
docs = docsearch.similarity_search(query)

print(docs[0].page_content)
