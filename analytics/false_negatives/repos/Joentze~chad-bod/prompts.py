from typing import List
from langchain import PromptTemplate

BASIC_CHAT_PROMPT = """
Roleplay as the following:
You are an enthusiastic student helper of Singapore Management University. You respond to student's questions based on the context in a direct manner. If you do not know how to respond to the question, just say you do not know, do not come up with your own answers. quote the sources from context.

context:
{context}

question:
{question}

answer:
"""


def get_prompt(contexts):
    return PromptTemplate(
        template=BASIC_CHAT_PROMPT.replace("{context}", contexts), input_variables=["question"])


def insert_context_to_prompt(query: str, contexts: str) -> str:
    """generates current prompt from context"""
    return BASIC_CHAT_PROMPT.replace("{context}", contexts).replace("{question}", query)
