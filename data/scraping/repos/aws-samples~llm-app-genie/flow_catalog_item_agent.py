""" Module that contains a class that represents an agent catalog item. """
from dataclasses import dataclass
from typing import Union

from langchain.chains.base import Chain

from .flow_catalog_item import FlowCatalogItem
from .retriever_catalog_item import RetrieverCatalogItem
from .catalog import CatalogById
from .model_catalog_item import ModelCatalogItem
from chatbot.llm_app import MRKLApp, SQLMRKLApp
from .agent_chain_catalog_item import AgentChainCatalogItem

from .agent_chain_catalog_item_sql_generator import AGENT_CHAIN_SQL_GENERATOR_NAME

AGENT_NAME = "[Experimental] Agents"


@dataclass
class AgentsItem(FlowCatalogItem):
    """
    Class that represents using an agent flow.
    """

    def __init__(self):
        super().__init__(AGENT_NAME)

    def enable_file_upload(self) -> bool:
        return False

    def enable_agents_chains(self) -> bool:
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
    ) -> Union[MRKLApp, SQLMRKLApp]:
        """
        Returns the llm app (i.e. agent in this case) to use for this flow.
        """
        
        if str(agent_chain)==AGENT_CHAIN_SQL_GENERATOR_NAME:
            agent_chain_type = "SQL"
        else:
            agent_chain_type = "TOOLS"

        chat_prompt = prompt_catalog[agent_chain.get_prompt_path()].get_instance()
        llm = model.get_instance()

        if agent_chain_type =="TOOLS":
            agent_chain = agent_chain.get_instance()
            return MRKLApp(
                prompt=chat_prompt,
                llm=llm,
                agent_chain=agent_chain
            )
        else:
            sql_llm  = sql_model.get_instance()
            return SQLMRKLApp(
                prompt=chat_prompt,
                llm=llm,
                sql_connection_uri=sql_connection_uri,
                sql_llm=sql_llm
            )
