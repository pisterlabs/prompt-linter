from pydantic import BaseModel
from core.config import settings
from core.log_config import logger

from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import AzureOpenAI
from langchain.retrievers import AzureCognitiveSearchRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

class Question(BaseModel):

    question: str

    def ask(self):

        # Create the retriever to get the documents from Azure Cognitive Services
        logger.debug("Creating retriever to get documents from Azure Cognitive Services")
        retriever = AzureCognitiveSearchRetriever(
            content_key = "content",
            service_name = settings.azure_cognitive_search_service_name,
            index_name = settings.azure_cognitive_search_index_name,
            api_key = settings.azure_cognitive_search_api_key
        )

        # Attempt to retrieve the documents from Azure Cognitive Services
        logger.debug("Retrieving documents from Azure Cognitive Services")
        try:
            docs = retriever.get_relevant_documents(self.question)
        except Exception as e:
            message = "Error retrieving documents from Azure Cognitive Search"
            logger.error(f"{message}: {e}")
            raise Exception(message) from e
        
        logger.info(f"Retrieved {len(docs)} documents from Azure Cognitive Search")

        # split the documents into chunks so they can be sent to OpenAI in Azure
        logger.debug("Splitting documents into chunks")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size = 500,
            chunk_overlap = 0
        )
        all_splits = splitter.split_documents(docs)

        # Create the embeddings to use for the question
        logger.debug("Creating embeddings for the question")
        embeddings = OpenAIEmbeddings(
            openai_api_key = settings.openai_api_key,
            openai_api_base = settings.openai_api_base,
            openai_api_version = settings.openai_api_version,
            openai_api_type = settings.openai_api_type,
            chunk_size = 16
        )

        # Create a vector db from the documents that have been retrieved
        logger.debug("Creating vector db from the documents")
        vectordb = FAISS.from_documents(
            documents = all_splits,
            embedding = embeddings
        )

        # Create the language model to use in the chain
        logger.debug("Create LLM using AzureOpenAI")
        llm  = AzureOpenAI(
            openai_api_key = settings.openai_api_key,
            openai_api_base = settings.openai_api_base,
            openai_api_version = settings.openai_api_version,
            openai_api_type = settings.openai_api_type,
            model_kwargs = {
                "engine": "text-davinci-003"
            }
        )

        # Create the AI chain to use to answer the question
        logger.debug("Creating AI chain to answer the question")
        chain = RetrievalQA.from_chain_type(
            llm = llm,
            retriever = vectordb.as_retriever(),
            chain_type = "stuff"
        )

        # Finally ask the question
        logger.debug("Ask the question of the AI chain")
        answer = chain.run(self.question)

        return answer