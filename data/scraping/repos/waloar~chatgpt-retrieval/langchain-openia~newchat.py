import os
import sys

import openai
import json
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader,DirectoryLoader,UnstructuredFileLoader,CSVLoader

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter, TokenTextSplitter
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

import constants

import registralogpapertrail as log 

def main(doc_location: str ='onepoint_chat'):
    """
    Main entry point for the application.
    It loads all texts from a specific folder and specific web pages, 
    creates the vector database and initializes the user interface.
    :param doc_location: The location of the CSV files
    """
    log.logger.info(f"Using doc location {doc_location}.")
    texts, doc_path = load_texts(doc_location=doc_location)
    website_texts = load_website_texts([
        'https://www.onepointltd.com/',
        'https://www.onepointltd.com/do-data-better/'
        ])
    texts.extend(website_texts)
    docsearch = extract_embeddings(texts=texts, doc_path=Path(doc_path))
    init_streamlit(docsearch=docsearch, texts=texts)


def load_texts(doc_location: str) -> tuple[list[str], Path]:
    """
    Loads the texts of the CSV file and concatenates all texts in a single list.
    :param doc_location: The document location.
    :return: a tuple with a list of strings and a path.
    """
    doc_path = Path(doc_location)
    texts = []
    for p in doc_path.glob("*.csv"):
        texts.extend(load_csv(p))
    log.logger.info(f"Length of texts: {len(texts)}")
    return texts, doc_path

def load_csv(file_path: Path) -> list[Document]:
    """
    Use the csv loader to load the CSV content as a list of documents.
    :param file_path: A CSV file path
    :return: the document list after extracting and splitting all CSV records.
    """
    loader = CSVLoader(file_path=str(file_path), encoding="utf-8")
    doc_list: list[Document] = loader.load()
    doc_list = [d for d in doc_list if d.page_content != 'Question: \nAnswer: ']
    log.logger.info(f"First item: {doc_list[0].page_content}")
    log.logger.info(f"Length of CSV list: {len(doc_list)}")
    return split_docs(doc_list)


def load_website_texts(url_list: list[str]) -> list[Document]:
    """
    Used to load website texts.
    :param url_list: The list with URLs
    :return: a list of documents
    """
    documents: list[Document] = []
    for url in url_list:
        text = text_from_html(requests.get(url).text)
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100, separator=".")
        texts = text_splitter.split_text(text)
        for t in texts:
            documents.append(Document(page_content=t))
    return documents

def extract_embeddings(texts: List[Document], doc_path: Path) -> Chroma:
    """
    Either saves the Chroma embeddings locally or reads them from disk, in case they exist.
    :return a Chroma wrapper around the embeddings.
    """
    embedding_dir = f"{cfg.chroma_persist_directory}/{doc_path.stem}"
    if Path(embedding_dir).exists():
        shutil.rmtree(embedding_dir, ignore_errors=True)
    try:
        docsearch = Chroma.from_documents(texts, cfg.embeddings, persist_directory=embedding_dir)
        docsearch.persist()
    except Exception as e:
        logger.error(f"Failed to process {doc_path}: {str(e)}")
        return None
    return docsearch


def init_streamlit(docsearch: Chroma, texts):
    """
    Creates the Streamlit user interface.
    This code expects some form of user question and as soon as it is there it processes
    the question.
    It can also process a question from a drop down with pre-defined questions.
    Use streamlit like this:
    streamlit run ./chat_main.py
    """
    title = "Ask questions about Onepoint"
    st.set_page_config(page_title=title)
    st.header(title)
    st.write(f"Context with {len(texts)} entries")
    simple_chat_tab, historical_tab = st.tabs(["Simple Chat", "Historical Questions"])
    with simple_chat_tab:
        user_question = st.text_input("Your question")
        with st.spinner('Please wait ...'):
            process_user_question(docsearch=docsearch, user_question=user_question)
    with historical_tab:
        user_question_2 = st.selectbox("Ask a previous question", read_history())
        with st.spinner('Please wait ...'):
            logger.info(f"question: {user_question_2}")
            process_user_question(docsearch=docsearch, user_question=user_question_2)

def process_user_question(docsearch: Chroma, user_question: str):
    """
    Receives a user question and searches for similar text documents in the vector database.
    Using the similar texts and the user question retrieves the response from the LLM.
    :param docsearch: The reference to the vector database object
    :param user_question: The question the user has typed.
    """
    if user_question:
        similar_docs: List[Document] = docsearch.similarity_search(user_question, k = 5)
        response, similar_texts = process_question(similar_docs, user_question)
        st.markdown(response)
        if len(similar_texts) > 0:
            write_history(user_question)
            st.text("Similar entries (Vector database results)")
            st.write(similar_texts)
        else:
            st.warning("This answer is unrelated to our context.")