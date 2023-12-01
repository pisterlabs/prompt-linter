from typing import List, Tuple
from langchain.docstore.document import Document
from pathlib import Path
from langchain.vectorstores import FAISS

import streamlit as st

import os

from doc_loader import load_txt
from config import cfg
from log_factory import logger

from dotenv import load_dotenv

from chat_factory import init_vector_search
from question_service import search_and_combine

load_dotenv()


def write_history(question):
    """
    Writes the question into a local history file.
    :param question: The text to be written to the local history file.
    """
    if len(question) > 0:
        with open(cfg.history_file, "a") as f:
            f.write(f"{question}\n")


@st.cache_data()
def read_history() -> List[str]:
    """
    Reads and caches some historical questions. Which you can use to ask questions in the UI.
    :return: a list of questions.
    """
    history_list = [""]
    with open(cfg.history_file, "r") as f:
        history_list.extend(list(set([l for l in f.readlines() if len(l.strip()) > 0])))
    return history_list


def process_user_question(
    docsearch: FAISS, user_question: str, chain_type: str, context_size: int
):
    """
    Receives a user question and searches for similar text documents in the vector database.
    Using the similar texts and the user question retrieves the response from the LLM.
    :param docsearch: The reference to the vector database object
    :param user_question: The question the user has typed.
    """
    if user_question:
        response, similar_texts, similar_metadata = search_and_combine(
            docsearch=docsearch,
            user_question=user_question,
            chain_type=chain_type,
            context_size=context_size,
        )
        st.markdown(response)
        if len(similar_texts) > 0:
            write_history(user_question)
            st.text("Similar entries (Vector database results)")
            st.write(similar_texts)
        else:
            st.warning("This answer is unrelated to our context.")


def create_chain_type(chain_key: str) -> str:
    """
    Determines the chain type (strategies to process documents) to use:
    map_reduce: It separates texts into batches (as an example, you can define batch size in llm=OpenAI(batch_size=5)), feeds each batch with the question to LLM separately, and comes up with the final answer based on the answers from each batch.
    refine : It separates texts into batches, feeds the first batch to LLM, and feeds the answer and the second batch to LLM. It refines the answer by going through all the batches.
    map-rerank: It separates texts into batches, feeds each batch to LLM, returns a score of how fully it answers the question, and comes up with the final answer based on the high-scored answers from each batch.
    chain_type="stuff" uses ALL of the text from the documents in the prompt. It actually doesn’t work with our example because it exceeds the token limit and causes rate-limiting errors
    """
    default_chain = "map_reduce"
    chain_type: str = st.selectbox(
        "Which chain type would you like to use?",
        (default_chain, "refine", "map_rerank", "stuff"),
        key=chain_key,
    )
    if chain_type is None:
        return default_chain
    return chain_type


def create_context_size_select(key: str) -> int:
    default_context_size = 5
    context_size: str = st.selectbox(
        "Which is the context size?",
        (str(default_context_size), "6", "7", "8", "9", "10"),
        key=key,
    )
    if context_size is None:
        return default_context_size
    return int(context_size)


def init_streamlit(docsearch: FAISS):
    """
    Creates the Streamlit user interface.
    This code expects some form of user question and as soon as it is there it processes
    the question.
    It can also process a question from a drop down with pre-defined questions.
    Use streamlit like this:
    streamlit run ./murli_chat.py
    """
    title = "Ask questions about the Murlis"
    st.set_page_config(page_title=title)
    st.header(title)
    # st.write(f"Context with {len(texts)} entries")
    simple_chat_tab, historical_tab = st.tabs(["Simple Chat", "Historical Questions"])
    with simple_chat_tab:
        chain_type = create_chain_type("question_chain_type")
        context_size = create_context_size_select("question_context_size")
        user_question = st.text_input("Your question")
        with st.spinner("Please wait ..."):
            process_user_question(
                docsearch=docsearch,
                user_question=user_question,
                chain_type=chain_type,
                context_size=context_size,
            )
    with historical_tab:
        chain_type = create_chain_type("historical_chain_type")
        user_question_2 = st.selectbox("Ask a previous question", read_history())
        if len(user_question_2) > 0:
            context_size = create_context_size_select("history_context_size")
            with st.spinner("Please wait ..."):
                logger.info(f"question: {user_question_2}")
                process_user_question(
                    docsearch=docsearch,
                    user_question=user_question_2,
                    chain_type=chain_type,
                    context_size=context_size,
                )


def load_texts(doc_location: str) -> Tuple[List[str], Path]:
    """
    Loads the texts of the CSV file and concatenates all texts in a single list.
    :param doc_location: The document location.
    :return: a tuple with a list of strings and a path.
    """
    doc_path = Path(doc_location)
    texts = []
    failed_count = 0
    for i, p in enumerate(doc_path.glob("*.txt")):
        try:
            logger.info(f"Processed {p}")
            texts.extend(load_txt(p))
        except Exception as e:
            logger.error(f"Cannot process {p} due to {e}")
            failed_count += 1
    logger.info(f"Length of texts: {len(texts)}")
    logger.warning(f"Failed: {failed_count}")
    return texts, doc_path


def main(doc_location: str = "onepoint_chat"):
    """
    Main entry point for the application.
    It loads all texts from a specific folder and specific web pages,
    creates the vector database and initializes the user interface.
    :param doc_location: The location of the CSV files
    """
    docsearch = init_vector_search()
    init_streamlit(docsearch=docsearch)


if __name__ == "__main__":
    main(os.environ["DOC_LOCATION"])
