""" Module that contains a class that represents a File Upload retriever catalog item. """
from dataclasses import dataclass

from langchain.chains.base import Chain

from .flow_catalog_item import FlowCatalogItem
from .retriever_catalog_item import RetrieverCatalogItem
from .catalog import CatalogById
from .model_catalog_item import ModelCatalogItem
from chatbot.llm_app import BaseLLMApp, LLMApp
from .agent_chain_catalog_item import AgentChainCatalogItem


UPLOAD_DOCUMENT_SEARCH = "Upload a document and search it"


@dataclass
class DocUploadItem(FlowCatalogItem):
    """
    Class that represents using a LLM without a retriever, but using an uploaded document.
    """

    def __init__(self):
        super().__init__(UPLOAD_DOCUMENT_SEARCH)

    def enable_file_upload(self) -> bool:
        return True

    def get_instance(self) -> Chain:
        return None

    def llm_app_factory(
        self, 
        model: ModelCatalogItem, 
        retriever: RetrieverCatalogItem, 
        agent_chain: AgentChainCatalogItem,
        prompt_catalog: CatalogById,
        sql_connection_uri: str,
        sql_model: ModelCatalogItem
    ) -> LLMApp:
        """
        Returns the llm app to use without a retriever, but using an uploaded document.
        """
        llm = model.get_instance()
        chat_prompt = prompt_catalog[model.chat_prompt_identifier].get_instance()

        return BaseLLMApp(prompt=chat_prompt, llm=llm)
