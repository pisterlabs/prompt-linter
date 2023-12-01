import os

from langchain.document_loaders import ReadTheDocsLoader
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
import pinecone
import glob

from dotenv import load_dotenv
load_dotenv()

pinecone.init(
    api_key=os.environ.get("PINECONE_API_KEY"),
    environment=os.environ.get("PINECONE_ENVIRONMENT_REGION"),
)
INDEX_NAME = "langchain-doc-index"


def ingest_docs():
    

# Specify the folder path and file pattern (e.g., '*.txt' for all text files)
    folder_path = '/home/leovanzei/projects/Langchain/viavi/pdf'
    file_pattern = '*.pdf'

    # Use glob to find files that match the pattern
    file_list = glob.glob(os.path.join(folder_path, file_pattern))

    # Iterate through the matched files
    for filepath in file_list:
        loader = PyPDFLoader(f"{filepath}")
        raw_documents = loader.load()
        print(f"loaded {len(raw_documents)} documents")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=50, separators=["\n\n", "\n", " ", ""]
        )
        documents = text_splitter.split_documents(raw_documents)
        for doc in documents:
            new_url = doc.metadata["source"]
            #new_url = new_url.replace("langchain-docs", "https:/")
            #doc.metadata.update({"source": new_url})

        embeddings = OpenAIEmbeddings()
        print(f"Going to add {len(documents)} to Pinecone")
        Pinecone.from_documents(documents, embeddings, index_name=INDEX_NAME)
        print("****Loading to vectorestore done ***")


if __name__ == "__main__":
    ingest_docs()
