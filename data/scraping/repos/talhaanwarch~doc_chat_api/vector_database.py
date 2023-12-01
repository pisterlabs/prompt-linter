from langchain.vectorstores import Milvus
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory.chat_message_histories.in_memory import ChatMessageHistory
from langchain.chat_models import ChatOpenAI
from utils import get_settings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from fastapi import HTTPException
import pymilvus
from langchain.llms import GPT4All

from langchain.schema import messages_from_dict
from utils import get_settings
from prompts import prompt_doc, prompt_chat

def vector_database(
              collection_name,
              drop_existing_embeddings=False,
              embeddings_name='sentence',
              doc_text=None):

    """
    Creates and returns a Milvus database based on the specified parameters.
    Args:
        doct_text: The document text. 
        collection_name: The name of the collection.
        drop_existing_embeddings: Whether to drop existing embeddings.
        embeddings_name: The name of the embeddings ('openai' or 'sentence').
    Returns:
        The Milvus database.
        """

    if embeddings_name == 'openai':
        embeddings = OpenAIEmbeddings(openai_api_key=get_settings().openai_api_key)
    elif embeddings_name == 'sentence':
        embeddings = HuggingFaceEmbeddings()
    else:
        print('invalid embeddings')
    if doc_text:
        try: 
            vector_db = Milvus.from_documents(
                doc_text,
                embeddings,
                collection_name=collection_name,
                drop_old=drop_existing_embeddings,
                connection_args={"host": get_settings().host, "port": "19530"},
                # if we want to communicate between two dockers then instead of local 
                # host we need to use milvus-standalone
            )
        except pymilvus.exceptions.ParamError:
            raise HTTPException(status_code=400,
                                detail=f"collection_name {collection_name} already exist. Either set drop_existing_embeddings to true or change collection_name")
    else:
        vector_db = Milvus(
            embeddings,
            collection_name=collection_name,
            connection_args={"host": get_settings().host, "port": "19530"},
        )
    return vector_db


def get_chat_history(inputs):
    """
    Get human input only
    """
    inputs = [i.content for i in inputs]
    # inputs = [string for index, string in enumerate(inputs) if index % 2 == 0]
    return '\n'.join(inputs)


def db_conversation_chain(llm_name, stored_memory, collection_name):

    """
    Creates and returns a ConversationalRetrievalChain based on the specified parameters.
    Args:
        llm_name: The name of the language model ('openai', 'gpt4all', or 'llamacpp').
        stored_memory: Existing conversation.
        collection_name: The name of the collection (optional).
    Returns:
        The ConversationalRetrievalChain.
    """

    if llm_name == 'openai':
        llm = ChatOpenAI(
            model_name='gpt-3.5-turbo',
            openai_api_key=get_settings().openai_api_key)  
        embeddings_name = 'openai'

    elif llm_name == 'gpt4all':
        llm = GPT4All(
            model='llms/ggml-gpt4all-j.bin', 
            n_ctx=1000, 
            verbose=True)
        embeddings_name = "sentence"

    elif llm_name == 'llamacpp':
        llm = GPT4All(
            model='llms/ggml-gpt4all-l13b-snoozy.bin', 
            n_ctx=1000, 
            verbose=True)
        embeddings_name = "sentence"

    vector_db = vector_database(
        collection_name=collection_name,
        embeddings_name=embeddings_name
        )

    if stored_memory:
        retrieved_messages = messages_from_dict(stored_memory)
        chat_history = ChatMessageHistory(messages=retrieved_messages)
    else:
        chat_history = ChatMessageHistory()
    memory = ConversationBufferMemory(memory_key="chat_history",
                                      return_messages=True,
                                      output_key='answer',
                                      chat_memory=chat_history
                                      )
    chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=vector_db.as_retriever(),
        memory=memory,
        chain_type="stuff",
        return_source_documents=True,
        verbose=True,
        condense_question_prompt=prompt_chat,
        return_generated_question=True,
        get_chat_history=get_chat_history,
        combine_docs_chain_kwargs={"prompt": prompt_doc}
        )
    return chain
