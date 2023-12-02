import logging
import os
from typing import Dict, Any

from langchain.chains import ConversationalRetrievalChain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA
from langchain.chains import RetrievalQAWithSourcesChain

from bedrock_api import get_conversation_memory

from constants import *

default_max_token_limit = os.environ["DEFAULT_MAX_TOKEN_LIMIT"]


logger = logging.getLogger()
logger.setLevel(logging.INFO)


def qa_from_langchain_and_vectorstore_v1(langchain_client, vectorstore, with_sources=True) -> Any:
    """
    integrate / tie together the vector store and actual LLM model's text (answer) generation
    
    note: uses RetrievalQAWithSourcesChain to get source documents 
          (result/output contains 'question', 'answer', etc.)
    """
    if with_sources:
        qa = RetrievalQAWithSourcesChain.from_chain_type(
            llm=langchain_client, 
            chain_type="stuff", 
            retriever=vectorstore.as_retriever(), 
            return_source_documents=True)
    else:
        qa = RetrievalQA.from_chain_type(
            llm=langchain_client, 
            chain_type="stuff", 
            retriever=vectorstore.as_retriever())
    return qa


def qa_from_langchain_and_vectorstore_v2(langchain_client, vectorstore, with_sources=True) -> Any:
    """
    integrate / tie together the vector store and actual LLM model's text (answer) generation
    
    note: uses RetrievalQA to get source documents 
          (result/output contains 'query', 'result', etc. (differs from above...))
    """
    if with_sources:
        qa = RetrievalQA.from_chain_type(
            llm=langchain_client, 
            chain_type="stuff", 
            retriever=vectorstore.as_retriever(), 
            return_source_documents=True,
            chain_type_kwargs={"prompt": ANTHROPIC_QA_PROMPT_TEMPLATE})
    else:
        qa = RetrievalQA.from_chain_type(
            llm=langchain_client, 
            chain_type="stuff", 
            retriever=vectorstore.as_retriever())
    return qa


def conv_qa_from_langchain_and_vectorstore(langchain_client, vectorstore, conversation_memory,
                                           chain_type='stuff', max_token_limit=default_max_token_limit, 
                                           verbose=False) -> Any:
    """
    contstruct conversational Q&A instance
    """
    memory_chain = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conv_qa = ConversationalRetrievalChain.from_llm(
        llm=langchain_client, 
        retriever=vectorstore.as_retriever(), 
        memory=conversation_memory,                                            
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
        verbose=verbose, 
        chain_type=chain_type,
        max_tokens_limit=max_token_limit
    )
    return conv_qa
    

# conv_qa cache keyed on 'session_id'
CONV_QA_CACHE = dict()

def fetch_conv_qa(session_id,
                  langchain_client, 
                  vectorstore,
                  chain_type='stuff', 
                  max_token_limit=default_max_token_limit,
                  reset_conversation=False,
                  verbose=False) -> Any:
    """
    either construct or fetch a qa_conv instance from cache
    each conv_qa has its own 'conversation memory' instance, thus allowing
    for session-specific conversations
    use 'reset_conversation=True' to force a new instance for the specified session id
    """
    global CONV_QA_CACHE
    
    conv_qa = None if reset_conversation else CONV_QA_CACHE.get(session_id)
    if not conv_qa:
        conversation_memory = get_conversation_memory()
        conv_qa = conv_qa_from_langchain_and_vectorstore(langchain_client, 
                                                         vectorstore, 
                                                         conversation_memory,
                                                         chain_type,
                                                         max_token_limit,
                                                         verbose)
        CONV_QA_CACHE[session_id] = conv_qa
    return conv_qa


def qa_from_langchain_and_vectorstore(langchain_client, vectorstore, with_sources=True) -> Any:
    """
    use v1 for now...
    """
    return qa_from_langchain_and_vectorstore_v1(langchain_client, vectorstore, with_sources)


def query_qa(qa, query) -> Dict[str, Any]:
    if type(qa) == RetrievalQAWithSourcesChain:
        result = qa(query)
        logger.info(f"result with sources: '{result}'")
        d = dict()
        d['question'] = result['question']
        d['answer'] = result['answer']
        source_documents = list()
        for doc in result['source_documents']:
            source_documents.append({
                'page_content': doc.page_content,
                'source': doc.metadata['source']
            })
        d['source_documents'] = source_documents
        result = d
    else:
        result = qa.run(query)
        result = dict(answer=result)
    return result


def query_conv(conv_qa, query) -> Dict[str, Any]:
    result = conv_qa.run({'question': query })
    result = dict(answer=result)
    return result


def reset_conv(conv_memory):
    conv_memory.clear()