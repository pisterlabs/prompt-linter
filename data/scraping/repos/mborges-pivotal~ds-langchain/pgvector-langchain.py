# Postgres Embedding
#
# see https://python.langchain.com/docs/integrations/vectorstores/pgembedding
#
import os
import getpass

os.environ["OPENAI_API_KEY"] = "<YOUR_API_KEY>"

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.docstore.document import Document

from langchain.vectorstores.pgvector import PGVector

loader = TextLoader("state_of_the_union.txt")

documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()

# PGVector needs the connection string to the database.
CONNECTION_STRING = "postgresql+psycopg2://marceloborges@localhost:5432/marceloborges"

db = PGVector.from_documents(
    embedding=embeddings,
    documents=docs,
    collection_name='langchain',
    connection_string=CONNECTION_STRING,
)

query = "What did the president say about Ketanji Brown Jackson"
docs_with_score = db.similarity_search_with_score(query)

for doc, score in docs_with_score:
    print("-" * 80)
    print("Score: ", score)
    print(doc.page_content)
    print("-" * 80)
