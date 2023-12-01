"""
RAG class uses conversational retriever from LangChain to query information from PDF document.
"""

import os
import openai
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA,  ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

openai.api_key  = os.environ['OPENAI_API_KEY']

class Rag:

    def __init__(self):
        self.llm_name = os.environ['LLM_NAME']
        self.file = os.environ['RAG_DOCUMENT']
        self.chain_type = os.environ['CHAIN_TYPE']
        self.k = int(os.environ['K_RAG'])
        self.qa = self.load_db(self.file, self.chain_type, self.k)

    def load_db(self, file, chain_type, k):
        """
        Read the PDF document, split it, storage as vectorstores and
        return a conversational retriever used to query into the document information.
        Args:
            :param file: PDF file path with return policies information.
            :param chain_type: chain type
            :param k:
        Returns:
            :returns qa: ConversationalRetrievalChain
        """
        # load PDF documents
        loader = PyPDFLoader(file)
        documents = loader.load()
        # split documents in chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        docs = text_splitter.split_documents(documents)
        # define embedding for the document
        embeddings = OpenAIEmbeddings()
        # create vector database from document data
        db = DocArrayInMemorySearch.from_documents(docs, embeddings)
        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})
        # create a chatbot chain which will be used to extract information
        qa = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(model_name=self.llm_name, temperature=0),
            chain_type=chain_type,
            retriever=retriever,
            return_source_documents=True,
            return_generated_question=True,
        )
        return qa

    def query(self, query, rag_retrieval_history, debug=True):
        """
        Query information into the conversational retriever.
        Args:
            :param query: user query.
            :param rag_retrieval_history: retrieval queries history
        Returns:
            :returns answer: string answer obtained with the conversational retriever
            :returns rag_retrieval_history: retrieval queries history
        """
        response = self.qa({"question": query, "chat_history": rag_retrieval_history})
        if debug:
            print(f'answer RAG: {response["answer"]}')
        rag_retrieval_history.extend([(query, response["answer"])])
        return response['answer'], rag_retrieval_history


if __name__ == "__main__":
    RAG = Rag()
    query = "What is Electronic bot return policies period"
    query = "how could I contact to Electronic bot for return policies"
    query = "What happen with defective items after 30 Days"
    answer = RAG.query_rag(query, [])
    print(answer)

