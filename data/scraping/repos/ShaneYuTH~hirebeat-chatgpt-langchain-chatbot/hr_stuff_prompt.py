# flake8: noqa
from langchain.chains.prompt_selector import ConditionalPromptSelector, is_chat_model
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

condense_template = """Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question, in its original language. Notably, if the user's intention, explicitly or implicitly, is for the AI to answer based solely on the chat history, the AI should strictly form a question that seeks a response from the chat history alone.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(condense_template)

prompt_template = """As an HR professional, use the provided context to answer the question at the end accurately. If the question specifically asks for information only from the chat history, limit your response to the chat history alone. Ensure that your answer is as relevant as possible to the given context. If you cannot provide a 100% accurate answer, offer the most relevant information you can, along with reasons for its relevance. If an answer is not relevant or you do not know the answer, kindly state that the information is not available.

{context}

Question: {question}
Helpful Answer:"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

system_template = """As an HR professional, use the following context to address the user's question.
If the question specifically asks for information only from the chat history, limit your response to the chat history alone. Ensure that your answer is as relevant as possible to the given context. If you cannot provide a 100% accurate answer, offer the most relevant information you can, along with reasons for its relevance. If an answer is not relevant or you do not know the answer, kindly state that the information is not available.
----------------
{context}"""
messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}"),
]
CHAT_PROMPT = ChatPromptTemplate.from_messages(messages)

PROMPT_SELECTOR = ConditionalPromptSelector(
    default_prompt=PROMPT, conditionals=[(is_chat_model, CHAT_PROMPT)]
)
