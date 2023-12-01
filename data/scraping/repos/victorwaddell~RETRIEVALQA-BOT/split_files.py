# Splitting the text into docs

from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_files(docs, chunk_size, chunk_overlap):
    print(f"Splitting files into chunks of size {chunk_size} with overlap {chunk_overlap}...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap)
    splits = text_splitter.split_documents(docs)
    print(f"Split files into {len(splits)} splits!\n")
    return splits