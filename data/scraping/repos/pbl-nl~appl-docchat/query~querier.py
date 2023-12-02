from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage
from langchain.vectorstores.chroma import Chroma
from loguru import logger
from langchain.llms import HuggingFaceHub
from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
# local imports
import settings
import utils as ut


class Querier:
    '''
    When parameters are read from settings.py, object is initiated without parameter settings
    When parameters are read from GUI, object is initiated with parameter settings listed
    '''
    def __init__(self, llm_type=None, llm_model_type=None, embeddings_provider=None, embeddings_model=None, 
                 vecdb_type=None, chain_name=None, chain_type=None, chain_verbosity=None, search_type=None, 
                 chunk_k=None, local_api_url=None):
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
        self.local_api_url = settings.API_URL if local_api_url is None and settings.API_URL is not None else local_api_url
        self.chat_history = []


    def make_agent(self, input_folder, vectordb_folder):
        """Create a langchain agent with selected llm and tools"""
        #
        # TODO
        #   - generalise code from make_chain
        #   - implement sample tools (wikipedia (standard), geocoder, soilgrids)
        #       see:
        #           https://python.langchain.com/docs/integrations/tools/wikipedia
        #           https://python.langchain.com/docs/integrations/tools/requests
        #           https://python.langchain.com/docs/modules/agents/tools/custom_tools (<-)
        #   - implement dynamic tool selection mechanism
        #   - create llm, tools, and initialise agent
        #   - add __init__ parameters for agent (maybe rename some chain related params?)
        #   - see usages of make_chain where to select between using chain and agent
        #   - add evaluation questions and answers, e.g. based on detailed spatial location context
        #
        return


    def make_chain(self, input_folder, vectordb_folder):
        self.input_folder = input_folder
        self.vectordb_folder = vectordb_folder

        if self.llm_type == "chatopenai":
            llm_model_type = "gpt-3.5-turbo"
            if self.llm_model_type == "gpt35_16":
                llm_model_type = "gpt-3.5-turbo-16k"
            elif self.llm_model_type == "gpt4":
                llm_model_type = "gpt-4"
            llm = ChatOpenAI(
                client=None,
                model=llm_model_type,
                temperature=0,
            )
        elif self.llm_type == "huggingface":
            # default value is llama-2, with maximum output length 512
            llm_model_type = "meta-llama/Llama-2-7b-chat-hf"
            max_length = 512
            if self.llm_model_type == 'GoogleFlan':
                llm_model_type = 'google/flan-t5-base'
                max_length = 512
            llm = HuggingFaceHub(repo_id=llm_model_type,
                                 model_kwargs={"temperature": 0.1,
                                               "max_length": max_length}
                                )
        elif self.llm_type == "local_llm":
            logger.info("Use Local LLM")
            logger.info("Retrieving " + self.llm_model_type)
            if self.local_api_url is not None: # If API URL is defined, use it
                logger.info("Using local api url " + self.local_api_url)
                llm = Ollama(
                    model=self.llm_model_type, 
                    base_url = self.local_api_url,
                    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
                )
            else:
                llm = Ollama(
                    model=self.llm_model_type,
                    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
                )
            logger.info("Retrieved " + self.llm_model_type)

        # get embeddings
        embeddings = ut.getEmbeddings(self.embeddings_provider, self.embeddings_model, self.local_api_url)

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
