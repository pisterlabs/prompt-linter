import streamlit as st
import os
import tempfile
from PIL import Image
import fitz
import os
import pandas as pd
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

from tools.tool_base import ToolBase
from const import DEMO_PATH, ENCODING_OPTIONS


class PdfChat(ToolBase):
    def __init__(self, logger):
        super().__init__(logger)
        self.title = "PDf-Chat"
        self.formats = ["Demo"]
        self.text = ""
        self.encoding_source = "utf-8"
        self.script_name, script_extension = os.path.splitext(__file__)
        self.intro = self.get_intro()

    def get_demo_documents(self):
        df = pd.read_csv((DEMO_PATH / "documents.csv"), sep=";")
        documents_dic = dict(zip(df["file_path"], df["title"]))
        return documents_dic

    def show_ui(self):
        doc_options = self.get_demo_documents()
        document = st.selectbox(
            label="Wähle ein Dokument aus",
            options=doc_options.keys(),
            format_func=lambda x: doc_options[x],
        )

        pdf_loader = PyPDFLoader(str(DEMO_PATH / document))
        documents = pdf_loader.load()

        text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        documents = text_splitter.split_documents(documents)
        vectordb = Chroma.from_documents(
            documents, embedding=OpenAIEmbeddings(), persist_directory="./data"
        )
        vectordb.persist()

        qa_chain = RetrievalQA.from_chain_type(
            llm=OpenAI(),
            retriever=vectordb.as_retriever(search_kwargs={"k": 7}),
            return_source_documents=True,
        )

        if "messages" not in st.session_state:
            st.session_state.messages = []

        if prompt := st.chat_input(
            f"Stelle eine Frage zum Dokument '{doc_options[document]}'"
        ):
            st.session_state.messages.append({"role": "user", "content": prompt})
            response = qa_chain({"query": prompt})
            st.session_state.messages.append({"role": "assistant", "content": response})

        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])
