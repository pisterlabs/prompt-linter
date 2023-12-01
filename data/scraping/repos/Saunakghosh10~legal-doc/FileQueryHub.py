import os
import streamlit as st
import pandas as pd
from docx import Document
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

def process_multiple_files(files):
    combined_text = ""
    for uploaded_file in files:
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        if file_extension == ".pdf":
            pdf_reader = PdfReader(uploaded_file)
            for page in pdf_reader.pages:
                combined_text += page.extract_text()
        elif file_extension == ".txt":
            combined_text += uploaded_file.read().decode("utf-8")
        elif file_extension == ".xlsx":
            excel_data = pd.read_excel(uploaded_file)
            combined_text += excel_data.to_string()
        elif file_extension == ".sql":
            combined_text += uploaded_file.read().decode("utf-8")
        elif file_extension == ".docx":
            doc = Document(uploaded_file)
            for paragraph in doc.paragraphs:
                combined_text += paragraph.text + "\n"
        elif file_extension == ".csv":
            csv_data = pd.read_csv(uploaded_file)
            combined_text += csv_data.to_string()
        # Add more file type handling here as needed
        else:
            st.warning(f"Unsupported file type: {file_extension}. Skipping.")
    # print(combined_text)
    return combined_text

def main():
    st.set_page_config(page_title="Declutter Legal Documents", page_icon="📄")
    st.header("Declutter Your Documents")

    files = st.file_uploader(
        "Upload multiple files",
        type=["pdf", "txt", "xlsx", "sql", "docx", "csv"],
        accept_multiple_files=True
    )

    if files:
        combined_text = process_multiple_files(files)
        # with st.expander("See explanation"):
            # st.write(combined_text)
        OPENAI_API_KEY = st.text_input("OPENAI API KEY", type="password")
    
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(combined_text)
        # st.write(chunks)
        # creating embeddings

        if OPENAI_API_KEY:
            embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
            # st.write("Embedding Created")
            # st.write(embeddings)
            with st.spinner("Creating Knowledge Base..."):
                knowledge_base = FAISS.from_texts(chunks, embeddings)
            st.success("Knowledge Base created")

            st.write("Chat with Multiple Files 🗣️📚")
            
            def ask_question(i=0):
                user_question = st.text_input("Ask a question about your PDF?",key = i)
                print(user_question)
                if user_question:
                    with st.spinner("Searching for answers..."):
                        docs = knowledge_base.similarity_search(user_question)
                        with st.expander("See docs"):
                            st.write(docs)

                        llm = OpenAI(openai_api_key=OPENAI_API_KEY)
                        chain = load_qa_chain(llm, chain_type="stuff")
                        with get_openai_callback() as cb:
                            response = chain.run(input_documents=docs, question=user_question)
                            print(cb)
                    st.write(response)
                    ask_question(i+1)
                    
            ask_question()

if __name__ == "__main__":
    main()