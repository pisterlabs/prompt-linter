from abc import ABC
import ssl
import os
from unstructured.partition.auto import partition
from unstructured.partition.xlsx import partition_xlsx
from unstructured.staging.base import convert_to_dict
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
import streamlit as st
import pandas as pd
from loguru import logger
import json
import chromadb
from dotenv import main
from chromadb.utils import embedding_functions
from langchain.callbacks import StreamlitCallbackHandler
import tempfile
import nltk

import fitz
import pandas as pd

from intriq.table_representation import table_to_docs

main.load_dotenv()
_PARTITION_STRATEGY = 'hi_res'
_PARTITION_MODEL_NAME = 'yolox'
_OPEN_AI_MODEL_NAME = 'gpt-3.5-turbo-0613'

# Download necessary NLTK data
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('stopwords')

st.set_page_config(
    page_title="intriq data chatbot", page_icon="🦜")
st.title("🔥🤖🔥 intriq data chatbot 🔥🤖🔥")


def clear_submit():
    """
    Clear the Submit Button State
    Returns:

    """
    st.session_state["submit"] = False


# drop files here
uploaded_files = st.file_uploader(
    "Drop all your shit here 💩",
    accept_multiple_files=True,
    help="Various File formats are supported",
    on_change=clear_submit,
)

if not uploaded_files:
    st.warning(
        "What am I a mind reader? Upload some data files if you want to chat with me."
    )

if uploaded_files:
    loaders = []
    for uploaded_file in uploaded_files:
        try:
            ext = os.path.splitext(uploaded_file.name)[1][1:].lower()
        except:
            ext = uploaded_file.split(".")[-1]
        if ext.lower() not in ['xlsx', 'xls']:
            logger.warning('not an excel file. Skipping')
            continue
        # convert to pdf
        pdf_file = None
        page = pdf_file[0]
        # for page in pdf_file:
        tabls = page.find_tables()
        for i, tabl in enumerate(tabls):
            logger.info(f'Table {i} column names: {tabl.header.names}')
        tabl_df = tabl[0].to_pandas()
        tabl_df = tabl_df.replace('\n', ' ', regex=True)

        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            for index, row in tabl_df.interrows():
                row_str = ''
                for col in tabl_df.columns:
                    row_str += f'{col}: {row[col]}'
                formatted = row_str[:-2]
                logger.info(formatted)
                tmp_file.write(formatted)

        loader = TextLoader(
            tmp_file.name,
            # encoding='utf-8'
        )
        docs = loader.load()
        text_splitter = CharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        chunks = text_splitter.split_documents(docs)
        embeddings = OpenAIEmbeddings()
        db = Chroma.from_documents(docs, embeddings)

        # try:
        #     ext = os.path.splitext(uploaded_file.name)[1][1:].lower()
        # except:
        #     ext = uploaded_file.split(".")[-1]
        # if ext.lower() not in ['xlsx', 'xls']:
        #     logger.warning('not an excel file. Skipping')
        #     continue
        # # convert to pdf file
        # pdf_file = uploaded_file  # change this
        # with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        #     tmp_file.write(pdf_file.getvalue())
        #     tmp_file_path = tmp_file.name
        # loader = UnstructuredPDFLoader(tmp_file_path)
        # loaders.append(loader)

    # index = VectorstoreIndexCreator().from_loaders(loaders)


if "messages" not in st.session_state or st.sidebar.button("Clear conversation history"):
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Whaddup? Ask me something about something. Or don't, that's fine.."}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input(placeholder="What is this data about?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    llm = ChatOpenAI(
        model_name=_OPEN_AI_MODEL_NAME,
        temperature=0,
        streaming=True
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=db.as_retriever()
    )

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(
            st.container(), expand_new_thoughts=False)
        docs = db.similarity_search(prompt)
        chain.run(
            {'query': prompt},
            callbacks=[st_cb]
        )
        # response = index.query_with_sources(prompt)
        # logger.info(response)
        st.session_state.messages.append(
            {"role": "assistant", "content": response})
        st.write(response)
