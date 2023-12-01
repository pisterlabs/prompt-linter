import sys
import os
sys.path.append(f'{os.path.dirname(__file__)}/../..')
from botcore.setup import trace_ai21
from langchain.chains import ConversationChain, LLMChain
from langchain.prompts import PromptTemplate

TEMPLATE = """ You are a secondhand dealer, an environmentalist.
You can answer many types of question about environemnt, recycling and secondhand product in general very well.
You are having a conversation with a user.
Based on your questions and user answers from the chat history.
 {chat_history}

 Given the question: {question}.
 Please give your best answer to the given question, along with an explanation for your answer."""


def build_bot_chat_chain(model, memory):
    prompt = PromptTemplate(input_variables=["question","chat_history"], template=TEMPLATE)
    chain = LLMChain(llm=model, prompt=prompt, memory=memory)
    return chain
