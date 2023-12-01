import os
import getpass

# read from .env file
from dotenv import load_dotenv
load_dotenv()

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import PGVector
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI


loader = TextLoader("./state_of_the_union.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()


pgpassword = os.getenv("PGPASSWORD", "")
if not pgpassword:
    pgpassword = getpass.getpass("Enter pgpassword: ")

CONNECTION_STRING = f"postgresql+psycopg2://pgadmin:{pgpassword}@pg-vec-geba.postgres.database.azure.com:5432/pgvector"

# ping postgress db
try:
    import sqlalchemy
    engine = sqlalchemy.create_engine(CONNECTION_STRING)
    engine.connect()
except Exception as e:
    print("Error connecting to postgres db")
    exit(1)

print("Connected to postgres db...")

COLLECTION_NAME = "state_of_the_union_test"

db = PGVector.from_documents(
    embedding=embeddings,
    documents=docs,
    collection_name=COLLECTION_NAME,
    connection_string=CONNECTION_STRING
)


retriever = db.as_retriever()

query = "What did the president say about Ketanji Brown Jackson"

qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=retriever)

answer = qa.run(query)

print(answer)

