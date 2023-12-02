from langchain.prompts.prompt import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain

from dotenv import load_dotenv

# Load the environment variables from the .env file
load_dotenv()

import os

_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.
You can assume the question about tfacebook.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

template = """You are an AI assistant for answering questions about facebook.
You are given the following extracted parts of a long document and a question. Provide a conversational answer.
If you don't know the answer, just say "Hmm, I'm not sure." Don't try to make up an answer.
If the question is not about facebook, politely inform them that you are asking out of context question.
Question: {question}
=========
{context}
=========
Answer in Markdown:"""
QA_PROMPT = PromptTemplate(template=template, input_variables=["question", "context"])


def get_chain(vectorstore):
    llm = OpenAI(temperature=0, openai_api_key=os.getenv('OPENAI_API_KEY'))
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        vectorstore.as_retriever(),
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
    )
    return qa_chain
