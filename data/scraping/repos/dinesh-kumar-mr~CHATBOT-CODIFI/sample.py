import streamlit as st
from dotenv import load_dotenv
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
from io import BytesIO

# Sidebar contents
with st.sidebar:
    st.title('🤗💬 LLM Chat App')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model
    ''')
    add_vertical_space(5)
    st.write('Made by Abinesh & Dinesh')

load_dotenv()

def main():
    st.header("Chat with PDF 💬")

    # Set default values for chunk_size and chunk_overlap.
    chunk_size = 1000
    chunk_overlap = 200

    # chunk_size = st.number_input('Set Chunk Size:', min_value=1, value=1000)
    # chunk_overlap = st.number_input('Set Chunk Overlap:', min_value=0, value=200)

    # Set the OpenAI API key as an environment variable
    os.environ["OPENAI_API_KEY"] = "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"

    # Dropdown to select a folder from a directory
    root_directory = 'Dataset'
    folders = [folder for folder in os.listdir(root_directory) if os.path.isdir(os.path.join(root_directory, folder))]
    selected_folder = st.selectbox("Select a folder from the list:", folders)
    folder_path = os.path.join(root_directory, selected_folder)

    # Dropdown to select PDF from the selected folder
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
    selected_pdf = st.selectbox("Select a PDF from the list:", pdf_files)
    pdf_path = os.path.join(folder_path, selected_pdf)

    if selected_pdf:
        with open(pdf_path, "rb") as f:
            pdf_bytes = BytesIO(f.read())
        pdf_reader = PdfReader(pdf_bytes)

        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
            )
        chunks = text_splitter.split_text(text=text)

        store_name = selected_pdf[:-4]

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
        else:
            embeddings = OpenAIEmbeddings()  # No need to pass API key since it's now an environment variable
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)

        query = st.text_input("Ask questions about your PDF file:")

        if query:
            docs = VectorStore.similarity_search(query=query, k=3)

            llm = OpenAI(model_name="gpt-3.5-turbo")  # Specify model here
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
            st.write(response)

if __name__ == '__main__':
    main()
