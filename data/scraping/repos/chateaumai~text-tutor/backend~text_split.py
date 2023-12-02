from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

#splitting book into chunks
def process_documents(page_contents):
    documents = [Document(page_content=content) for content in page_contents]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1200,
        chunk_overlap  = 50
    )
    split_documents = text_splitter.split_documents(documents)
    return split_documents