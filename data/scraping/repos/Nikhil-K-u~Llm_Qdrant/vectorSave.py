from langchain.vectorstores import Qdrant
from langchain.embeddings.openai import OpenAIEmbeddings
import qdrant_client
import os
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
def get_pdf_text(pdf_file):
    text =""
    for pdf in pdf_file:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
         text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
       separator="\n",
       chunk_size=1000,
       chunk_overlap=200,
       length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def main():
    load_dotenv()
    st.set_page_config(page_title="Save user book vertex to the Qdrant cloud")
    pdf_file = st.file_uploader("upload the pdf" ,
        accept_multiple_files=True
    )
    client = qdrant_client.QdrantClient(
     os.getenv("QDRANT_HOST"),
     api_key=os.getenv("QDRANT_API_KEY")
    )
    vectors_config = qdrant_client.http.models.VectorParams(
        size=1536,
        distance=qdrant_client.http.models.Distance.COSINE)

    client.recreate_collection(
        collection_name=os.getenv("QDRANT_COLLECTION_NAME"),
        vectors_config = vectors_config,
    )
    embedding =OpenAIEmbeddings()
    vector_store = Qdrant(
        client=client,
        collection_name=os.getenv("QDRANT_COLLECTION_NAME"), 
        embeddings=embedding
    )
    if st.button("Submit"):
       with st.spinner("Processing"):
            text = get_pdf_text(pdf_file)
            chunks = get_text_chunks(text)
            vector_store.add_texts(chunks)
            



if __name__ == "__main__":
    main()