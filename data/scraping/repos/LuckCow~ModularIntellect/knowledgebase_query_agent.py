import pickle
import os

from langchain import OpenAI, LLMChain, FAISS
from langchain.callbacks import StdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.chains import ChatVectorDBChain
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import BaseLLM
from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.components.itask import ITask
from src.services.chainlang_agent_service import BaseChainLangAgent
from langchain.chains.chat_vector_db.prompts import (CONDENSE_QUESTION_PROMPT,
                                                     QA_PROMPT)

from src.visualization.vector_search import visualize_vector_search
from src.web.socketio_callbackmanager import SocketIOCallbackHandler


def ingest_docs(knowledge_path: str, storage_path: str):
    """Get documents from repository."""
    loader = DirectoryLoader(knowledge_path, loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"},
                             recursive=True, silent_errors=True)
    raw_documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    documents = text_splitter.split_documents(raw_documents)
    embeddings = OpenAIEmbeddings(allowed_special=["<|endoftext|>"])
    vectorstore = FAISS.from_documents(documents, embeddings)

    # Save vectorstore
    with open(storage_path, "wb") as f:
        pickle.dump(vectorstore, f)

    return vectorstore


class SimpleKnowledgeBaseQueryAgent(BaseChainLangAgent):
    """Agent that queries a knowledge base."""

    def __init__(self, llm: BaseLLM, knowledge_path: str, storage_path: str="vectorstore.pkl", socketio=None):
        # set llm (from dependency injection)
        self.llm = llm

        # check if storage_path exists
        if not os.path.exists(storage_path):
            print("Ingesting knowledge to create vectorstore")
            self.vectorstore = ingest_docs(knowledge_path, storage_path)
        else:
            with open(storage_path, "rb") as f:
                self.vectorstore = pickle.load(f)

        self.socketio = socketio

        # initialize chat history
        self.chat_history = []

        super().__init__()

    def _get_chain(self):
        """Create a ChatVectorDBChain for question/answering."""
        # Construct a ChatVectorDBChain with a streaming llm for combine docs
        # and a separate, non-streaming llm for question generation

        manager = CallbackManager([])
        manager.set_handler(StdOutCallbackHandler())
        if self.socketio:
            manager.add_handler(SocketIOCallbackHandler(self.socketio, 'ChatVectorDBChain'))

        question_generator = LLMChain(
            llm=self.llm, prompt=CONDENSE_QUESTION_PROMPT, callback_manager=manager, verbose=True
        )
        doc_chain = load_qa_chain(
            self.llm, chain_type="stuff", prompt=QA_PROMPT, callback_manager=manager, verbose=True
        )

        qa = ChatVectorDBChain(
            vectorstore=self.vectorstore,
            combine_docs_chain=doc_chain,
            question_generator=question_generator,
        )
        return qa

    def execute(self, task: ITask):

        #visualize_vector_search(self.vectorstore, task)
        result = self._chain({"question": task, "chat_history": self.chat_history})
        self.chat_history.append((task, result["answer"]))
        return result['answer']
