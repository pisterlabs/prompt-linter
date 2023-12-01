import os
from typing import Dict

import streamlit as st
from dotenv import load_dotenv
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import GoogleDriveLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.neo4j_vector import Neo4jVector
from streamlit.logger import get_logger

from chains import (
    load_embedding_model,
    load_llm,
)

load_dotenv(".env")

SCOPES = ["https://www.googleapis.com/auth/drive"]

url = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")
ollama_base_url = os.getenv("OLLAMA_BASE_URL")
embedding_model_name = os.getenv("EMBEDDING_MODEL")
llm_name = os.getenv("LLM")
# Remapping for Langchain Neo4j integration
os.environ["NEO4J_URL"] = url

logger = get_logger(__name__)

embeddings, dimension = load_embedding_model(
    embedding_model_name, config={"ollama_base_url": ollama_base_url}, logger=logger
)


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


llm = load_llm(llm_name, logger=logger, config={"ollama_base_url": ollama_base_url})


def load_credentials():
    creds = service_account.Credentials.from_service_account_file(
        'credentials.json', scopes=SCOPES
    )
    return creds


def load_folders(service) -> Dict[str, str]:
    """
    This function will prompt the user to select a folder from his google drive
    :return: folder id
    """

    options = {}
    try:
        folders = service.files().list(q="mimeType='application/vnd.google-apps.folder'",
                                       fields='files(id, name)').execute()
        for i, folder in enumerate(folders['files']):
            options[folder['name']] = folder['id']
    except HttpError as error:
        print(f"An error occurred: {error}")

    return options


if "messages" not in st.session_state:
    st.session_state.messages = []


def main():
    st.header("📄Chat with your google drive folder")
    creds = load_credentials()
    with st.sidebar:
        st.title("Drive searcher")
        st.write('Share the folder in your google drive with the account')
        st.write(f'Share your folder with: {creds.service_account_email}')
        st.write('Then select the folder in the sidebar')
        if True:
            service = build("drive", "v3", credentials=creds)
            folders = load_folders(service)
            selected_folder = st.sidebar.selectbox('Choose a folder', list(folders.keys()),
                                                   key='selected_folder')
            if selected_folder:
                folder_id = folders[selected_folder]

    loader = GoogleDriveLoader(
        folder_id=folder_id,
        recursive=False,
        service_account_key="credentials.json"
    )
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000, chunk_overlap=0, separators=[" ", ",", "\n"]
    )

    texts = text_splitter.split_documents(docs)

    chunks = text_splitter.split_documents(documents=texts)

    # Store the chunks part in db (vector)
    vectorstore = Neo4jVector.from_documents(
        chunks,
        url=url,
        username=username,
        password=password,
        embedding=embeddings,
        index_name="pdf_bot",
        node_label="PdfBotChunk",
        pre_delete_collection=True,  # Delete existing PDF data
    )

    qa_chain = load_qa_chain(llm, chain_type="stuff")

    query = st.text_input("Ask questions about related your files in the folder")

    if query:
        docs = vectorstore.similarity_search(query)
        stream_handler = StreamHandler(st.empty())
        qa_chain.run(input_documents=docs, question=query, callbacks=[stream_handler])


if __name__ == "__main__":
    main()
