from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

CHROMA_PERSIST_DIRECTORY = "./.chroma"

load_dotenv()

if __name__ == "__main__":
    loader = TextLoader("./app.py")
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    db = Chroma.from_documents(
        docs, embeddings, persist_directory=CHROMA_PERSIST_DIRECTORY
    )
    db.persist()
