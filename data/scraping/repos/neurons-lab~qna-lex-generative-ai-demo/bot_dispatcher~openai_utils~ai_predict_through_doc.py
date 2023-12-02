#!/usr/bin/env python
import os
import sys
sys.path.append(os.path.dirname(__file__))

import boto3
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder
)

# from openai_utils.kendra_index_retriever import KendraIndexRetriever
from kendra_index_retriever import KendraIndexRetriever
# from openai_utils.api_utils import get_openai_api_key
from api_utils import get_openai_api_key


# AWS Region
region_name = os.getenv('AWS_REGION', 'us-east-1')
# Set SSM Parameter Store name for the OpenAI API key and the OpenAI Model Engine
API_KEY_PARAMETER_PATH = os.getenv('API_KEY_PARAMETER_PATH', '/openai/api_key')
# Kendra Index ID
KENDRA_INDEX_ID = os.getenv('KENDRA_INDEX_ID', '5a8804f9-be66-4468-8d2b-6db46ccd3ee9')

# Create an SSM client using Boto3
ssm = boto3.client('ssm', region_name=region_name)

openai_api_key=get_openai_api_key(ssm_client=ssm, parameter_path=API_KEY_PARAMETER_PATH)

retriever = KendraIndexRetriever(kendraindex=KENDRA_INDEX_ID,
    awsregion=region_name,
    return_source_documents=True
)

llm = OpenAI(temperature=0, openai_api_key=openai_api_key)


def qa_with_memory():
    """QA with Memory
    Example:
    >>> query = "Did he mention who she suceeded"
        result = qa({"question": query})
    
    """
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    qa = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=retriever,
        memory=memory,
        verbose=True,
    )
    return qa


def qa_with_pass_chat_history():
    """QA with Pass Chat History
    Example:
    >>> chat_history = [(query, result["answer"])]
        query = "Did he mention who she suceeded"
        result = qa({"question": query, "chat_history": chat_history})
    """
    qa = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=retriever,
        verbose=True,
    )
    return qa


def qa_with_pass_chat_history_multi_model():
    """QA with Pass Chat History
    Example:
    >>> chat_history = [(query, result["answer"])]
        query = "Did he mention who she suceeded"
        result = qa({"question": query, "chat_history": chat_history})
    """
    qa = ConversationalRetrievalChain.from_llm(
        ChatOpenAI(
            temperature=0,
            model="gpt-3.5-turbo-16k",
            openai_api_key=openai_api_key,
            max_tokens=2000
        ),
        retriever=retriever,
        verbose=True,
        condense_question_llm = ChatOpenAI(
            temperature=0,
            model='gpt-3.5-turbo-16k',
            openai_api_key=openai_api_key
        ),
    )
    return qa


def qa_with_condense_question():
    question_generator = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT)
    doc_chain = load_qa_chain(
        llm,
        chain_type="map_rerank",
        verbose=True,
        )
    chain = ConversationalRetrievalChain(
        retriever=retriever,
        question_generator=question_generator,
        combine_docs_chain=doc_chain,
        #verbose=True,
    )
    return chain


class AIPredictThroughDoc(object):
    def __init__(self):
        self.qa = qa_with_pass_chat_history_multi_model()

    def predict(self, query, chat_history=[]):
        result = self.qa({"question": query, "chat_history": chat_history})
        return result


if __name__ == '__main__':

    chat_history = []
    query = "How to secure deploy application in AWS account?"
    qa = qa_with_pass_chat_history_multi_model()
    result = qa({"question": query, "chat_history": chat_history})
    chat_history = [(query, result["answer"])]
    query = "My application is serverless"
    result = qa({"question": query, "chat_history": chat_history})
    print(result["answer"])
