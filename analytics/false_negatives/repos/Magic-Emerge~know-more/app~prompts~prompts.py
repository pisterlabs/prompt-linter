# flake8: noqa
# 翻译来自：from langchain.chains.chat_vector_db.prompts import (CONDENSE_QUESTION_PROMPT, QA_PROMPT)
from langchain.prompts.prompt import PromptTemplate

_template = """给定以下聊天记录和后续输入问题，将后续输入问题改写为独立问题。
聊天记录:
{chat_history}
后续输入问题: {question}
独立问题:
"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

prompt_template = """使用以下内容来回答最后的问题。如果你不知道答案，就回答你不知道，不要试图编造答案。
{context}
问题: {question}
答案:
"""
QA_PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)
