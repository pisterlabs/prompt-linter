# flake8: noqa
from langchain.prompts import PromptTemplate

custom_qa_template = """ You are a specialist in Youtube videos question and answers.
You are a slack bot, your role is to try to help regarding a provided youtube video.
Given the following extracted parts of the complete video transcript, the video meta data and a question, create a final answer. 
If you don't know the answer, just say that you don't know.

QUESTION: {question}
=========
PARTS:
=========
{context}
META DATA:
=========
{meta}
=========
FINAL ANSWER:"""
CUSTOM_YT_PROMPT = PromptTemplate(template=custom_qa_template, input_variables=["context", "question", "meta"])

CUSTOM_YT_EXAMPLE_PROMPT = PromptTemplate(
    template="Content: {page_content}",
    input_variables=["page_content"],
)