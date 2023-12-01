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

os.environ["OPENAI_API_KEY"] = ""

class ChatBotLoader:

    def load_db(file, chain_type, k):
        #openai.api_key = self.api_key_handler.get_openai_api_key()
        
        # Load documents
        loader = PyPDFLoader(file)
        documents = loader.load()
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        docs = text_splitter.split_documents(documents)
        
        # Define embedding
        embeddings = OpenAIEmbeddings()
        
        # Create vector database from data
        db = DocArrayInMemorySearch.from_documents(docs, embeddings)
        
        # Define retriever
        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})
        
        # Create a chatbot chain. Memory is managed externally.
        qa = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(model_name="gpt-3.5-turbo-0301", temperature=0),
            chain_type=chain_type,
            retriever=retriever,
            return_source_documents=True,
            return_generated_question=True,
        )
        return qa

