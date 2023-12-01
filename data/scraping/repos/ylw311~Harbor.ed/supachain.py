import os

from langchain.chat_models import ChatCohere
# from langchain.llms import Cohere
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.embeddings import CohereEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.pydantic_v1 import BaseModel
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import SupabaseVectorStore
from supabase.client import Client, create_client

# Conversational Chat
from operator import itemgetter
from langchain.schema import format_document
from langchain.schema.runnable import RunnableMap
from langchain.memory import ConversationBufferMemory


from env import COHERE_API_KEY, COHERE_MODEL, INPUT_TYPE, COHERE_EMBEDDINGS, TEMPERATURE, SUPABASE_URL, SUPABASE_SERVICE_KEY, CHUNK_SIZE, CHUNK_OVERLAP

memory = ConversationBufferMemory(
    return_messages=True, output_key="answer", input_key="question"
)


supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
TABLE_NAME = "documents"
FUNC_NAME = "match_documents"

# LLM:
# model = Cohere(cohere_api_key=COHERE_API_KEY,
#                model=COHERE_MODEL, temperature=TEMPERATURE)
model = ChatCohere(cohere_api_key=COHERE_API_KEY,
                   streaming=False, #model=COHERE_MODEL,
                   temperature=TEMPERATURE)

# Embedder
embeddings = CohereEmbeddings(
    cohere_api_key=COHERE_API_KEY, model=COHERE_EMBEDDINGS, truncate="END")

# Read from Supabase
vectorstore = SupabaseVectorStore(
    client=supabase_client,
    embedding=embeddings,
    table_name=TABLE_NAME,
    query_name=FUNC_NAME,
)
retriever = vectorstore.as_retriever()

# Chat History Processors
DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")
def _combine_documents(
    docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)


from typing import List, Tuple
def _format_chat_history(chat_history: List[Tuple]) -> str:
    buffer = ""
    for dialogue_turn in chat_history:
        human = "Human: " + dialogue_turn[0]
        ai = "Assistant: " + dialogue_turn[1]
        buffer += "\n" + "\n".join([human, ai])
    return buffer

# RAG prompt
## Condense question prompt
_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

## Answer prompt
template = """Answer the question based only on the following context:
{context}
Question: {question}
"""
ANSWER_PROMPT = ChatPromptTemplate.from_template(template)

# Memory
loaded_memory = RunnablePassthrough.assign(
    chat_history=RunnableLambda(memory.load_memory_variables) | itemgetter("history"),
)

# Now we calculate the standalone question
standalone_question = {
    "standalone_question": {
        "question": lambda x: x["question"],
        "chat_history": lambda x: _format_chat_history(x["chat_history"]),
    }
    | CONDENSE_QUESTION_PROMPT
    | model
    | StrOutputParser(),
}

# Get RAG documents
retrieved_documents = {
    "docs": itemgetter("standalone_question") | retriever,
    "question": lambda x: x["standalone_question"],
}

# Build Final Prompt
final_inputs = {
    "context": lambda x: _combine_documents(x["docs"]),
    "question": itemgetter("question"),
}

# Answer
answer = {
    "answer": final_inputs | ANSWER_PROMPT | model,
    "docs": itemgetter("docs"),
}
# And now we put it all together!
chain = loaded_memory | standalone_question | retrieved_documents | answer


def _ingest(url: str) -> dict:
    # Load docs
    # loader = PyPDFLoader(url)
    loader = TextLoader(url)
    data = loader.load()

    # Split docs
    print("Splitting documents")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    # text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(data)

    # Insert the documents in Supabase
    _ = SupabaseVectorStore.from_documents(
        docs,
        embeddings,
        client=supabase_client,
        table_name=TABLE_NAME,
        query_name=FUNC_NAME,
        chunk_size=CHUNK_SIZE,
    )

    return {}


ingest = RunnableLambda(_ingest)


def _similarity(query: str, k: int = 20):
    results = retriever.similarity_search(
        query=query,
        k=k,
    )
    return results


similarity = RunnableLambda(_similarity)
