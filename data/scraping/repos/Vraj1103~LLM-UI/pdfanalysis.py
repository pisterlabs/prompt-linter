import streamlit as st
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os

# from pdf_store import storePDF

# from llama_index import VectorStoreIndex
# from llama_hub.file.pdf.base import PDFReader
# from ll

with st.sidebar:
    st.title("Isolated Falcons")
    st.markdown('''
    # About
    This is a web application that uses the GPT-3 API to generate summaries of text.
    ''')
    add_vertical_space(5)

def main():
    st.header("PDF Analysis")
    pdf = st.file_uploader("Upload a PDF", type='pdf')
    # st.write(pdf.name)
    # st.write(pdf.type)

    if pdf is not None:
        # qe, ctx = storePDF()
        st.write(pdf.name)
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)

        store_name = pdf.name[:-4]

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl","rb") as f:
                VectorStore = pickle.load(f)
            st.write("Loaded existing store")
        else:
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)
            
            st.write("Embeddings computation complete")

        query = st.text_input("Ask questions related to the uploaded file", key="input")
        # st.write(query)

        if query:
            docs = VectorStore.similarity_search(query=query,k=3)
            # st.write(docs)
            llm = OpenAI()
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print (cb)
            st.write(response)


        # st.write(chunks)
        # st.write(text)
        # st.write(pdf_reader)


if __name__ == '__main__':
    main()