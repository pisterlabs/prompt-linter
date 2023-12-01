""" Module that contains a class that represents No search retriever catalog item. """
from dataclasses import dataclass

from langchain.chains.base import Chain

from .flow_catalog_item import FlowCatalogItem
from .retriever_catalog_item import RetrieverCatalogItem
from .catalog import CatalogById
from .model_catalog_item import ModelCatalogItem
from chatbot.llm_app import BaseLLMApp, LLMApp
from .agent_chain_catalog_item import AgentChainCatalogItem


SIMPLE_CHATBOT = "Only Chat"


@dataclass
class SimpleChatFlowItem(FlowCatalogItem):
    """
    Class that represents using a LLM without a retriever.
    """

    def __init__(self):
        super().__init__(SIMPLE_CHATBOT)

    def enable_file_upload(self) -> bool:
        return False

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
        Returns the llm app to use without retriever.
        """

        llm = model.get_instance()
        chat_prompt = prompt_catalog[model.chat_prompt_identifier].get_instance()

        return BaseLLMApp(prompt=chat_prompt, 
                        llm=llm, 
                        )
