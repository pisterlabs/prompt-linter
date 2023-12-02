import os
from dotenv import load_dotenv, find_dotenv

import pinecone

from langchain.chat_models import ChatCohere
from langchain.embeddings import CohereEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Pinecone
from typing import Any, Dict, List


def llm_call(query: str, chat_history: List[Dict[str, Any]] = []):
    _ = load_dotenv(find_dotenv(), override=True)

    pinecone.init(
        api_key=os.environ["PINECONE_API_KEY"],
        environment=os.environ["PINECONE_ENVIRONMENT"],
    )

    embeddings = CohereEmbeddings(model="embed-english-v3.0")

    docsearch = Pinecone.from_existing_index(
        embedding=embeddings,
        index_name=os.environ["PINECONE_INDEX"],
    )

    chat = ChatCohere(
        verbose=True,
        temperature=0,
    )

    qry = ConversationalRetrievalChain.from_llm(
        llm=chat, retriever=docsearch.as_retriever(), return_source_documents=True
    )
    return qry({"question": query, "chat_history": chat_history})
