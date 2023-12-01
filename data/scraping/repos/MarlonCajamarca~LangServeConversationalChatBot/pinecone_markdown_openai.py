import os
import argparse
import time
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone

from dotenv import load_dotenv
load_dotenv()

PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX", "langchain-test")

def main(input_directory:str = "simetrik-knowledge-bot/datasets/test_Academy_UR"):
    # Load all markdown files in the input directory
    loader = DirectoryLoader(
        path=input_directory, 
        glob="**/*.md",
        loader_cls=TextLoader,
        use_multithreading=True)
    documents = loader.load()

    # Chunking Strategy - Split the markdown files on the headers
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3")
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    
    # Split the documents into chunks following the chunking strategy
    document_chunks = []
    for document in documents:
        markdown_header_splits = markdown_splitter.split_text(document.page_content)
        for split in markdown_header_splits:
            split.metadata["source"] = document.metadata["source"]
            document_chunks.append(split)

    # Embed the chunks and add them to the vectorstore
    vectorstore = Pinecone.from_documents(
        documents=document_chunks,
        embedding=OpenAIEmbeddings(),
        index_name=PINECONE_INDEX_NAME
        )


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", 
                        type=str, 
                        default="simetrik-knowledge-bot/datasets/test_Academy_UR")
    args = parser.parse_args()

    # Run and time main function in charge of reading the markdown files and populating the vectorstore
    start = time.time()
    main(input_directory=args.input_dir)
    end = time.time()

    # Print the time in seconds it took to read, chunk, embed and populate the vectorstore
    print(f"Time to populate vectorstore: {end-start} seconds")

    # Print message saying that the vectorstore was created and populated successfully
    print(f"Pinecone Vectorstore Index {PINECONE_INDEX_NAME} populated successfully")
