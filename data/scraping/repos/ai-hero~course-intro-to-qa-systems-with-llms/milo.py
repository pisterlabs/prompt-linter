"""A basic chatbot using the OpenAI API + Community Notion Info"""
import logging
import os
import sys
from typing import Any, Dict, Generator, List, Tuple, Union

import openai
import streamlit as st
from answer_generator import answer_question
from chat_db_helper import ChatVectorDB
from dotenv import load_dotenv

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

ResponseType = Union[Generator[Any, None, None], Any, List, Dict]

# Load the .env file
load_dotenv()

# Set up the OpenAI API key
assert os.getenv("OPENAI_API_KEY"), "Please set your OPENAI_API_KEY environment variable."
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load the chat vector database
chat_vector_db = ChatVectorDB()


def extract_answer(summaries: List[str], question: str) -> Any:
    """Combine the summaries into a prompt and use SotA GPT-4 to answer."""
    context = "\n".join(summaries)
    return answer_question(question=question, context=context)


def get_answer(question: str) -> Tuple[str, Any]:
    """Get the answer to the question."""
    # Get answer to the question by finding the three conversations that are nearest
    # to the question and then using them to generate the answer.
    print("Searching documents nearest to the question.")
    chats = chat_vector_db.search_index(question)
    summaries = "\n".join([chat.summary for chat in chats])
    return summaries, extract_answer([chat.summary for chat in chats], question)


def main() -> None:
    """Run the chatbot."""
    st.title("Ask Milo from the MLOps.Community Chats👋")
    if not os.getenv("OPENAI_API_KEY"):
        st.error("Please set your OPENAI_API_KEY environment variable.")
        st.stop()

    question = st.text_input(
        "Enter your question",
        placeholder="What are some of the best models for tabular data?",
    )
    if not question:
        st.error("Please enter a question.")
        st.stop()

    if question:
        # Streamlit is progressive, everytime you change something in the UI,
        # code below will be re-run
        with st.spinner("Thinking..."):
            summaries, answer = get_answer(question)
            st.write("### Answer")
            st.write(answer)
            st.write("")
            st.write("### Summaries")
            st.markdown(summaries)


if __name__ == "__main__":
    main()
