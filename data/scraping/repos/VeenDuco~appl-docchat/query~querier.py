from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import AIMessage, HumanMessage
from langchain.vectorstores.chroma import Chroma
from loguru import logger
# local imports
import settings


class Querier:
    # When parameters are read from settings.py, object is initiated without parameter settings
    # When parameters are read from GUI, object is initiated with parameter settings listed
    def __init__(self, llm_type=None, llm_model_type=None, embeddings_provider=None, embeddings_model=None, 
                 vecdb_type=None, chain_name=None, chain_type=None, chain_verbosity=None, search_type=None, chunk_k=None):
        load_dotenv()
        self.llm_type = settings.LLM_TYPE if llm_type is None else llm_type
        self.llm_model_type = settings.LLM_MODEL_TYPE if llm_model_type is None else llm_model_type
        self.embeddings_provider = settings.EMBEDDINGS_PROVIDER if embeddings_provider is None else embeddings_provider
        self.embeddings_model = settings.EMBEDDINGS_MODEL if embeddings_model is None else embeddings_model
        self.vecdb_type = settings.VECDB_TYPE if vecdb_type is None else vecdb_type
        self.chain_name = settings.CHAIN_NAME if chain_name is None else chain_name
        self.chain_type = settings.CHAIN_TYPE if chain_type is None else chain_type
        self.chain_verbosity = settings.CHAIN_VERBOSITY if chain_verbosity is None else chain_verbosity
        self.search_type = settings.SEARCH_TYPE if search_type is None else search_type
        self.chunk_k = settings.CHUNK_K if chunk_k is None else chunk_k
        self.chat_history = []


    def make_chain(self, input_folder, vectordb_folder):
        self.input_folder = input_folder
        self.vectordb_folder = vectordb_folder

        if self.llm_type == "chatopenai":
            if self.llm_model_type == "gpt35":
                llm_model_type = "gpt-3.5-turbo"
            elif self.llm_model_type == "gpt35_16":
                llm_model_type = "gpt-3.5-turbo-16k"
            elif self.llm_model_type == "gpt4":
                llm_model_type = "gpt-4"
            llm = ChatOpenAI(
                client=None,
                model=llm_model_type,
                temperature=0,
            )

        if self.embeddings_provider == "openai":
            embeddings = OpenAIEmbeddings(model=self.embeddings_model, client=None)
            logger.info("Loaded openai embeddings")

        if self.vecdb_type == "chromadb":
            vector_store = Chroma(
                collection_name=self.input_folder,
                embedding_function=embeddings,
                persist_directory=self.vectordb_folder,
            )
            retriever = vector_store.as_retriever(search_type=self.search_type, search_kwargs={"k": self.chunk_k})
            logger.info(f"Loaded chromadb from folder {self.vectordb_folder}")


        if self.chain_name == "conversationalretrievalchain":
            self.chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                chain_type=self.chain_type,
                verbose=self.chain_verbosity,
                return_source_documents=True
            )

        logger.info("Executed Querier.make_chain(self, input_folder, vectordb_folder)")


    def ask_question(self, question: str):
        logger.info(f"current chat history: {self.chat_history}")
        response = self.chain({"question": question, "chat_history": self.chat_history})
        logger.info(f"question: {question}")
        answer = response["answer"]
        logger.info(f"answer: {answer}")
        self.chat_history.append(HumanMessage(content=question))
        self.chat_history.append(AIMessage(content=answer))
        return response
    
    def clear_history(self):
        # used by "Clear Conversation" button
        self.chat_history = []
