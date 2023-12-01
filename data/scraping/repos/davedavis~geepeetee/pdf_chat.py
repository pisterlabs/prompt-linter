"""This app uses Streamlit to accept a PDF, split it into chunks, create
embeddings and then use the ChatGPT API to query the doc based on the
embedding search.

Usage: streamlit run pdf_chat.py
"""

from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain import FAISS, OpenAI
from langchain.callbacks import get_openai_callback
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter


def main():
    # Initial Setup.
    load_dotenv()
    st.set_page_config(page_title="Dave's GPT PDF Query Tool")
    st.header("PDF Query Tool 💀")
    pdf = st.file_uploader("Upload PDF", type="pdf")

    # Extract text from PDF
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # split into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        chunks = text_splitter.split_text(text)

        # create embeddings
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        # show user input
        user_question = st.text_input("❓ Ask a question about the PDF:")
        if user_question:
            docs = knowledge_base.similarity_search(user_question)

            llm = OpenAI()
            chain = load_qa_chain(llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs,
                                     question=user_question)
                print(cb)

            st.write(response)



if __name__ == '__main__':
    main()