from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from dotenv import load_dotenv
from scraper import django_docs_build_urls

load_dotenv()


CHROMA_DB_DIRECTORY = "chroma_db/ask_django_docs"


def build_database():
    # We are using the function that's defined in scraper.py
    urls = django_docs_build_urls()

    # We can do the scraping ourselves and only look for .docs-content
    loader = WebBaseLoader(urls)
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    splitted_documents = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()

    db = Chroma.from_documents(
        splitted_documents,
        embeddings,
        collection_name="ask_django_docs",
        persist_directory=CHROMA_DB_DIRECTORY,
    )
    db.persist()
    print("worked")
    
build_database()