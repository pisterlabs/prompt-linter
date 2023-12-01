import os
import openai
import pickle
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter


st.header("Chat with PDF 💬")
# Sidebar contents
with st.sidebar:
    st.title('Munir PDF-Reader')
    st.markdown('''
    ## About
    Hey there! I'm an AI-powered PDF tool ready to extract meaning from documents and have conversations about the content. Just upload a file and you can query me just like you would query a human who read the material. Give it a try!
    ''')
    add_vertical_space(26)
    st.header(
        'Extension of Munir-GPT')


load_dotenv()
API_KEY = st.secrets["OPENAI_API_KEY"]


def main():
    # Upload the PDF File
    pdf = st.file_uploader("Upload a PDF file", type=["pdf"])

    # Read the PDF file
    if pdf is not None:
        pdf_reader = PdfReader(pdf)

        # Extract the text from the PDF
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        chunks = text_splitter.split_text(text=text)

        # Embeddings
        store_name = pdf.name[:-4]

        # Check if the store exists
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
        else:  # Create the store
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)

        # Ask questions
        query = st.text_input("Ask questions about your document:")

        if query:
            docs = VectorStore.similarity_search(query=query, k=3)
            # Load the Open AI Language Model
            llm = OpenAI(model_name='gpt-3.5-turbo')
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)
            st.write(response)


if __name__ == '__main__':
    main()
