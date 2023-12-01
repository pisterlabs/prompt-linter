import os
from dotenv import load_dotenv
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
import openai


load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

PERSIST_DIRECTORY = 'web_db'

def load_and_split_documents():
    documents = []
    loader = WebBaseLoader(['https://www.appwizardpro.com', 'https://www.appwizardpro.com/services','https://www.appwizardpro.com/contact-us', 'https://www.appwizardpro.com/about-us'])
    loaded_docs = loader.load_and_split()
    documents.extend(loaded_docs)

    return documents


def embed_documents(documents):
    embedding = OpenAIEmbeddings()
    vector_db = Chroma.from_documents(documents, embedding, persist_directory=PERSIST_DIRECTORY)
    vector_db.persist()


def main():
    documents = load_and_split_documents()

    embed_documents(documents)

    print("We finished loading and embedding the documents.", documents)

if __name__ == "__main__":
    main()
