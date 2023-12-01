"""
Purpose:
    AWS Well-Architected Chatbot
"""

# Python imports
from typing import Type, Union, Dict, Any, List
import os
import pickle

# 3rd party imports
import streamlit as st
import utils
import openai
import pandas as pd


openai.api_key = os.environ["OPEN_AI_KEY"]


@st.cache_data
def load_data_frame(fname):
    df = pd.read_csv(fname)
    return df


@st.cache_data
def load_embeddings(fname):
    with open(fname, "rb") as file:
        # Call load method to deserialze
        document_embeddings = pickle.load(file)

    return document_embeddings


def sidebar() -> None:
    """
    Purpose:
        Shows the side bar
    Args:
        N/A
    Returns:
        N/A
    """

    st.sidebar.image(
        "https://d1.awsstatic.com/gamedev/Programs/OnRamp/gt-well-architected.4234ac16be6435d0ddd4ca693ea08106bc33de9f.png",
        use_column_width=True,
    )

    st.sidebar.markdown(
        "AWS Well-Architected helps cloud architects build secure, high-performing, resilient, and efficient infrastructure for a variety of applications and workloads. Built around six pillars—operational excellence, security, reliability, performance efficiency, cost optimization, and sustainability—AWS Well-Architected provides a consistent approach for customers and partners to evaluate architectures and implement scalable designs.\n This chatbot is a **prototype** application and is for demonstration purposes only."
    )


def app() -> None:
    """
    Purpose:
        Controls the app flow
    Args:
        N/A
    Returns:
        N/A
    """

    # Spin up the sidebar
    sidebar()
    # Load questions
    query = st.text_input("Query:")

    df = load_data_frame("min_aws_wa.csv")
    document_embeddings = load_embeddings("document_embeddings.pkl")

    if st.button("Submit Query"):
        with st.spinner("Generating..."):
            answer, docs = utils.get_answer_from_chatgpt(
                query,
                df,
                document_embeddings,
            )

            st.markdown(answer)

            st.subheader("Resources")
            for doc in docs:
                st.write(doc)


def main() -> None:
    """
    Purpose:
        Controls the flow of the streamlit app
    Args:
        N/A
    Returns:
        N/A
    """

    # Start the streamlit app
    st.title("AWS Well-Architected Chatbot")
    st.subheader("Ask and Learn")
    st.warning("This is a prototype application, do not enter in confidential information.")
    
    app()


if __name__ == "__main__":
    main()
