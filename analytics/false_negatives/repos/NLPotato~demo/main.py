__import__("pysqlite3")

import sys, io, os

if "pysqlite3" in sys.modules:
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

from modules.base import BaseBot
from modules.templates import (
    PDF_PREPROCESS_TEMPLATE,
    PDF_PREPROCESS_TEMPLATE_WITH_CONTEXT,
)
from modules.preprocessors import PDFPreprocessor

import chromadb
from pprint import pprint
from langchain.document_loaders import PDFPlumberLoader
from langchain.text_splitter import MarkdownTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import streamlit as st
from streamlit import session_state as sst

PATH = "./user_data.pdf"
COLLECTION_NAME = "woori_pdf_prev_md"


@st.cache_resource
def init_bot():
    client = chromadb.PersistentClient("db/chroma/woori_pdf_prev_md")
    # collection = client.get_collection(name=COLLECTION_NAME)
    vectorstore = Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=OpenAIEmbeddings(),
    )
    bot = BaseBot(
        vectorstore=vectorstore,
    )
    # if not os.path.exists(path):
    #     bot = BaseBot.from_new_collection(
    #         loader=PDFPlumberLoader(path),
    #         preprocessor=PDFPreprocessor(
    #             prompt=PDF_PREPROCESS_TEMPLATE_WITH_CONTEXT,
    #         ),
    #         collection_name=COLLECTION_NAME,
    #     )
    # else:
    #     DB_DIR = "db/chroma/"
    #     client_settings = chromadb.config.Settings(
    #         chroma_db_impl="duckdb+parquet",
    #         persist_directory=DB_DIR,
    #         anonymized_telemetry=False,
    #     )
    #     embeddings = OpenAIEmbeddings()

    #     vectorstore = Chroma(
    #         collection_name=COLLECTION_NAME,
    #         embedding_function=embeddings,
    #         client_settings=client_settings,
    #         persist_directory=DB_DIR,
    #     )
    #     bot = BaseBot(
    #         vectorstore=vectorstore,
    #         collection_name=COLLECTION_NAME,
    #     )
    print(bot)
    return bot


@st.cache_data
def get_info():
    return """

    """


st.title("GPT-Powered Chat Bot")
info = get_info()


if "messages" not in sst:
    sst.messages = []

# uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
uploaded_file = 1
if uploaded_file is not None:
    # with open(PATH, "wb") as f:
    #     f.write(uploaded_file.read())

    if "bot" not in sst:
        sst.bot = init_bot()

    for message in sst.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("무엇이든 물어보세요"):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        sst.messages.append({"role": "user", "content": prompt})

        # Get assistant response
        response = sst.bot(
            prompt
        )  # keys: [question, chat_history, answer, source_documents]
        print(response)
        answer = response["answer"]
        source = response["source_documents"][0]
        source_content = source.page_content  # .replace("\n", " ")
        source_src = source.metadata["source"]
        source_page = source.metadata["page"]

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(answer)
            if "죄송" not in answer or "정보가 제공되지" not in answer or "찾을 수 없" in answer:
                # output = io.StringIO()
                # pprint(source_content, stream=output)
                # output_string = output.getvalue()
                st.divider()
                st.info(f"주요 출처 페이지: {source_page}")
                st.markdown(source_content)
                # st.info(f"출처 문서: {output_string}\n\n출처 링크: {source_src}")
            # Add assistant response to chat history
            sst.messages.append({"role": "assistant", "content": answer})
