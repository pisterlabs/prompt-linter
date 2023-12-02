"""UI utilities for the app."""

import streamlit as st
from app.utils.vector_database import connect_to_vector_db
from dotenv import find_dotenv, load_dotenv
from langchain.embeddings import HuggingFaceBgeEmbeddings
from pymilvus import connections

# import langchain
# from langchain.cache import SQLiteCache

state = st.session_state


def set_state_if_absent(key, value):
    """Set the state if it is absent."""
    if key not in state:
        state[key] = value


def init_session_states():
    """Some session states, that should always be present."""
    set_state_if_absent(key="rows", value=2)
    set_state_if_absent(
        key="embedding_model",
        value=HuggingFaceBgeEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            encode_kwargs={"normalize_embeddings": True},
            query_instruction="Represent this sentence for searching relevant "
            "passages: ",
        ),
    )
    set_state_if_absent(key="topic_model_fitted", value=False)


def start_app():
    """Start the app.

    Connect to Milvus default, set app_state to initialized and load .env file.
    """
    # langchain.llm_cache = SQLiteCache(database_path=".app.db")
    milvus_connection = {"alias": "default", "host": "localhost", "port": 19530}
    connections.connect(**milvus_connection)
    load_dotenv(find_dotenv())
    state["app_state"] = "initialized"


def reset_app():
    """Reset the app."""
    for key in state:
        del state[key]
    init_session_states()


def choose_collection(collections: list[str], on_change=None):
    """Choose a collection from a list of collections."""
    collections = sorted([c.replace("_", " ").title() for c in collections])
    collection_name = st.selectbox(
        "Select a research collection:",
        options=collections,
        index=0,
        on_change=on_change,
    )
    collection_name = collection_name.replace(" ", "_").lower()
    state["collection_name"] = collection_name
    state["nice_collection_name"] = state["collection_name"].replace("_", " ").title()
    connect_to_vector_db()


def display_vector_db_info():
    """If the vector database is loaded, display relevant info."""
    if "vector_db" in state:
        st.write(
            f"Total abstracts in collection: {state['vector_db'].col.num_entities} \n\n"
            "Embedding Dimensions: "
            f"{state['vector_db'].col.schema.fields[-1].params['dim']} \n\n"
        )


def sidebar_collection_info() -> None:
    """Display info about the current collection in the sidebar."""
    info = (
        f"Currently connected to collection: **{state['nice_collection_name']}**."
        if "collection_name" in state
        else "Currently you are not connected to any collection. "
    )
    st.sidebar.info(info)
