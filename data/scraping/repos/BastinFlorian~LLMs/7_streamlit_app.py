# Streamlit
import streamlit as st

# Imports
# Env var
import os
import sys
from dotenv import load_dotenv, find_dotenv

# Langchain
import openai
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings

# Compressor
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# Prompt template
from langchain.prompts import PromptTemplate

# Conversational chain
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Streaming
from langchain.callbacks.streaming_stdout_final_only import (
    FinalStreamingStdOutCallbackHandler,
)

from utils import load_docs_from_jsonl

# Global Variables
PATH_NAME_SPLITTER = './splitted_docs.jsonl'
PERSIST_DIRECTORY = 'docs/chroma/'

# Env variables
sys.path.append('../..')
_ = load_dotenv(find_dotenv())
openai.api_key = os.environ['OPENAI_API_KEY']


@st.cache_resource
def load_chain_depencdencies():
    # Embedding model
    embedding = OpenAIEmbeddings()

    # LLM
    llm = OpenAI(temperature=0, callbacks=[FinalStreamingStdOutCallbackHandler()])

    # Documents
    docs = load_docs_from_jsonl(PATH_NAME_SPLITTER)

    return docs, embedding, llm


@st.cache_resource
def get_vector_db(_embedding, _docs):
    # Chroma
    persist_directory = 'docs/chroma/'
    vectordb = Chroma.from_documents(
        documents=_docs,
        embedding=_embedding,
        persist_directory=persist_directory
    )

    return vectordb


@st.cache_resource
def get_compressor(_llm, _vectordb):
    # Compressor
    compressor = LLMChainExtractor.from_llm(_llm)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=_vectordb.as_retriever()
    )
    return compression_retriever


# Prompt
template = """
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Keep the answer as concise as possible.
You can answer with bullet point as well.

Always say "Merci pour cette question!" at the end of the answer.


CONTEXT: {context}
-------
HISTORY:
{chat_history}
Human: {question}
Assistant:

"""

QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question", "chat_history"], template=template)


@st.cache_resource
def get_chain(_llm, _vectordb, _qa_chain_prompt=QA_CHAIN_PROMPT):

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=_llm,
        retriever=_vectordb.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": _qa_chain_prompt},
        verbose=True,
        rephrase_question=False
    )

    return qa_chain


docs, embedding, llm = load_chain_depencdencies()
vectordb = get_vector_db(embedding, docs)
compression_retriever = get_compressor(llm, vectordb)
qa_chain = get_chain(llm, vectordb)

# Streamlit
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Comment puis-je vous renseigner ?"}]

if "history" not in st.session_state:
    st.session_state["history"] = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})

    st.chat_message("user").write(prompt)
    result = qa_chain({"question": prompt})

    # Ueseful for QA to get history
    st.session_state["history"].append((prompt, result["answer"]))

    # Useful to display messages in streamlit
    msg = {"role": "assistant", "content": result["answer"]}
    st.session_state.messages.append(msg)

    st.chat_message("assistant").write(result["answer"])
