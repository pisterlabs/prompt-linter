""" 
Custom Langchain tool to ask human
"""

import time

import streamlit as st
from langchain.tools.base import BaseTool


class CustomAskHumanTool(BaseTool):
    """Tool that asks user for input."""

    name = "AskHuman"
    description = """Use this tool if you don't find a specific answer using KendraRetrievalTool.\
Ask the human to clarify the question or provide the missing information.\
The input should be a question for the human."""

    def _run(
        self,
        query: str,
        run_manager=None,
    ) -> str:
        if "user_answer" not in st.session_state:
            answer_container = st.chat_message("assistant", avatar="🦜")
            answer_container.write(query)

            answer = st.text_input("Enter your answer", key="user_answer")
            while answer == "":
                time.sleep(1)

        return st.session_state["user_answer"]
