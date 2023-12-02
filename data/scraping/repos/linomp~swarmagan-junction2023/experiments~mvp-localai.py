import os
import sys

from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema.vectorstore import VectorStoreRetriever
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

load_dotenv()

CHAT_MODEL = "gpt-3.5-turbo"
EMBEDDINGS_MODEL = "text-embedding-ada-002"

EMBEDDINGS_PERSIST_DIR = 'experiments/db'

if "--realapi" in sys.argv:
    # NB: api key should be present in top-level .env file to use the real api
    OPENAI_API_BASE = "https://api.openai.com/v1"
else:
    OPENAI_API_BASE = "http://localhost:8080/v1"
    os.environ["OPENAI_API_KEY"] = "KEY_DOESNT_MATTER_FOR_LOCALHOST"

os.environ["OPENAI_API_BASE"] = OPENAI_API_BASE


def create_store():
    # Load and process the text
    loader = TextLoader('data/state_of_the_union_short.txt')
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=70)
    texts = text_splitter.split_documents(documents)

    # Embed and store the texts
    # Supplying a persist_directory will store the embeddings on disk
    embedding = OpenAIEmbeddings(model=EMBEDDINGS_MODEL, openai_api_base=OPENAI_API_BASE)
    vectordb = Chroma.from_documents(documents=texts, embedding=embedding, persist_directory=EMBEDDINGS_PERSIST_DIR)

    vectordb.persist()


def test_query_with_retrieval():
    # Load and process the text
    embedding = OpenAIEmbeddings(model=EMBEDDINGS_MODEL, openai_api_base=OPENAI_API_BASE)

    # Now we can load the persisted database from disk, and use it as normal.
    llm = ChatOpenAI(temperature=0, model_name=CHAT_MODEL, openai_api_base=OPENAI_API_BASE)
    vectordb = Chroma(persist_directory=EMBEDDINGS_PERSIST_DIR, embedding_function=embedding)
    retriever = VectorStoreRetriever(vectorstore=vectordb)
    qa = RetrievalQA.from_llm(llm=llm, retriever=retriever)

    query = "What the president said about taxes ?"
    print(qa.run(query))


def test_basic_query():
    llm = ChatOpenAI(temperature=0, model_name=CHAT_MODEL, openai_api_base=OPENAI_API_BASE)
    print(llm.call_as_llm("Who are you?"))


if __name__ == "__main__":
    if not os.path.exists(EMBEDDINGS_PERSIST_DIR):
        print("Creating embeddings...")
        create_store()

    test_query_with_retrieval()
