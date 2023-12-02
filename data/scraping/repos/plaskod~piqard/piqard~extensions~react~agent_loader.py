import os

import ruamel.yaml

from piqard.extensions.react.action import Action
from piqard.extensions.react.agent import Agent
from piqard.information_retrievers.google_custom_search import GoogleCustomSearch
from piqard.information_retrievers.faiss_retriever import FAISSRetriever
from piqard.information_retrievers.BM25_retriever import BM25Retriever
from piqard.information_retrievers.annoy_retriever import AnnoyRetriever
from piqard.information_retrievers.wiki_api import WikiAPI
from piqard.language_models.bloom_176b_api import BLOOM176bAPI
from piqard.language_models.cohere_api import CohereAPI
from piqard.language_models.gpt_j6b_api import GPTj6bAPI
from piqard.utils.prompt_template import PromptTemplate


class AgentLoader:
    """
    AgentLoader is a class that loads an Agent object from a YAML file or a string.
    """

    def __init__(self):
        self.yaml = ruamel.yaml.YAML()
        self.yaml.register_class(FAISSRetriever)
        self.yaml.register_class(BM25Retriever)
        self.yaml.register_class(AnnoyRetriever)
        self.yaml.register_class(GoogleCustomSearch)
        self.yaml.register_class(BLOOM176bAPI)
        self.yaml.register_class(GPTj6bAPI)
        self.yaml.register_class(CohereAPI)
        self.yaml.register_class(PromptTemplate)
        self.yaml.register_class(Action)
        self.yaml.register_class(WikiAPI)

    def load(self, config: str) -> Agent:
        """
        Loads an Agent object from a YAML file or a string.

        :param config: The YAML file or string to load.
        :return: The loaded Agent object.
        """
        if os.path.isfile(config):
            with open(config, "r") as f:
                return Agent(**self.yaml.load(f))
        else:
            return Agent(**self.yaml.load(config))
