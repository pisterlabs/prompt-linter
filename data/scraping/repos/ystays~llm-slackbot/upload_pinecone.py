from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pinecone 
import os
from dotenv import load_dotenv
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language

from langchain.document_loaders.sitemap import SitemapLoader


def init_with_dir_and_upload_vectors(embeddings, file_dir='./output'):
    loader = DirectoryLoader(file_dir, glob='**/*.txt', show_progress=True)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    docsearch = Pinecone.from_documents(docs, embeddings, index_name=os.environ['PINECONE_INDEX_NAME'])

def init_with_website_and_upload_vectors(embeddings):
    loader = SitemapLoader("https://saatva.com/sitemap.xml")
    documents = loader.load()
    html_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.HTML, chunk_size=60, chunk_overlap=0)
    html_docs = html_splitter.split_documents(documents)
    docsearch = Pinecone.from_documents(html_docs, embeddings, index_name=os.environ['PINECONE_INDEX_NAME'])

if __name__ == "__main__":
    load_dotenv()
    embeddings = OpenAIEmbeddings()

    # initialize pinecone
    pinecone.init(
        api_key=os.environ['PINECONE_API_KEY'],  # find at app.pinecone.io
        environment=os.environ['PINECONE_ENV']  # next to api key in console
    )

    n = input("Do you want to load the documents and upload the vectors to Pinecone? (y/n)")
    if n == 'y':
        init_with_dir_and_upload_vectors(embeddings)
    # init_with_website_and_upload_vectors(embeddings)