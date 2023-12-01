# import
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader

# load the document and split it into chunks
loader = TextLoader("../../../state_of_the_union.txt")
documents = loader.load()

# split it into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# create the open-source embedding function
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# load it into Chroma
db = Chroma.from_documents(docs, embedding_function)

# query it
query = "What did the president say about Ketanji Brown Jackson"
docs = db.similarity_search(query)

# print results
print(docs[0].page_content)

#--------------------
# save to disk
db2 = Chroma.from_documents(docs, embedding_function, persist_directory="./chroma_db")
docs = db2.similarity_search(query)

# load from disk
db3 = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)
docs = db3.similarity_search(query)
print(docs[0].page_content)

#--------------------
import chromadb

persistent_client = chromadb.PersistentClient()
collection = persistent_client.get_or_create_collection("collection_name")

langchain_chroma = Chroma(
    client=persistent_client,
    collection_name="collection_name",
    embedding_function=embedding_function,
)

print("There are", langchain_chroma._collection.count(), "in the collection")

#--------------------
# create the chroma client
import chromadb
import uuid
from chromadb.config import Settings

client = chromadb.HttpClient(settings=Settings(allow_reset=True))
collection = client.create_collection("my_collection")
for doc in docs:
        ids=[str(uuid.uuid1())], metadatas=doc.metadata, documents=doc.page_content
    )

# tell LangChain to use our client and collection name
db4 = Chroma(client=client, collection_name="my_collection", embedding_function=embedding_function)
query = "What did the president say about Ketanji Brown Jackson"
docs = db4.similarity_search(query)
print(docs[0].page_content)

#--------------------
# create simple ids
ids = [str(i) for i in range(1, len(docs) + 1)]

# add data
example_db = Chroma.from_documents(docs, embedding_function, ids=ids)
docs = example_db.similarity_search(query)
print(docs[0].metadata)

# update the metadata for a document
docs[0].metadata = {
    "source": "../../../state_of_the_union.txt",
    "new_value": "hello world",
}
print(example_db._collection.get(ids=[ids[0]]))

# delete the last document
print("count before", example_db._collection.count())
print("count after", example_db._collection.count())

#--------------------
# get a token: https://platform.openai.com/account/api-keys

from getpass import getpass
from langchain.embeddings.openai import OpenAIEmbeddings

OPENAI_API_KEY = getpass()

#--------------------
import os

#--------------------
embeddings = OpenAIEmbeddings()
new_client = chromadb.EphemeralClient()
openai_lc_client = Chroma.from_documents(
    docs, embeddings, client=new_client, collection_name="openai_collection"
)

query = "What did the president say about Ketanji Brown Jackson"
docs = openai_lc_client.similarity_search(query)
print(docs[0].page_content)

#--------------------
docs = db.similarity_search_with_score(query)

#--------------------
docs[0]

#--------------------
retriever = db.as_retriever(search_type="mmr")

#--------------------


#--------------------
# filter collection for updated source

#--------------------
