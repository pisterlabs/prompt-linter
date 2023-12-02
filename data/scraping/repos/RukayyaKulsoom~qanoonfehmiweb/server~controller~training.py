import os
import sys
import pinecone
from langchain.vectorstores import Pinecone
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import openai
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import TextLoader

os.environ["OPENAI_API_KEY"] = "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"
embeddings = OpenAIEmbeddings(deployment="text-embedding-ada-002")
def split_docs(documents,chunk_size=2000,chunk_overlap=1000):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  docs = text_splitter.split_documents(documents)
  return docs
def main():
    # Get the content from the command-line argument
    textContent = sys.argv[1] if len(sys.argv) > 1 else "Default content"

    # Split the document
    docs=[]
    file_path = "/Users/ramisha/Desktop/fyp2/qanoonfehmiweb/server/uploaded_file.txt"
    loader = TextLoader(file_path)
    document = loader.load()
    # Split the document
    docs = split_docs(document)

    pinecone.init(
        api_key="230a9212-b4a7-49fe-9c32-34ccdb7fbe40",  # find at app.pinecone.io
        environment="gcp-starter" , # next to api key in console
    )
    index_name = "train"
    index = Pinecone.from_documents(docs, embeddings, index_name=index_name)
if __name__ == "__main__":
    main()
