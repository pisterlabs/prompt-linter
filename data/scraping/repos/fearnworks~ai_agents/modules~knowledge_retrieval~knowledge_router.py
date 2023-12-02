from loguru import logger
from modules.knowledge_retrieval.domains.business_domain import BusinessChain, get_business_chain_config
from modules.knowledge_retrieval.domains.family_domain import FamilyChain, get_family_chain_config
from modules.knowledge_retrieval.domains.gaming_domain import GamingChain, get_gaming_chain_config
from modules.knowledge_retrieval.domains.finance_domain import FinanceChain, get_finance_chain_config
from modules.base.llm_chain_config import LLMChainConfig
from modules.knowledge_retrieval.base.router_chain import RouterChain
from modules.knowledge_retrieval.destination_chain import DestinationChainStrategy
from langchain import PromptTemplate, LLMChain

import pprint
from modules.settings.user_settings import UserSettings
from typing import Callable, Dict, Optional, Tuple
import re

class KnowledgeDomainRouter(RouterChain):
    def __init__(self, config: LLMChainConfig, question: str, display: Callable):
        settings = UserSettings.get_instance()
        
        chains : Dict[int, DestinationChainStrategy] = {
        1: BusinessChain(config=get_business_chain_config(), display=display),
        2: FamilyChain(config=get_family_chain_config(), display=display),
        3: GamingChain(config=get_gaming_chain_config(), display=display),
        4: FinanceChain(config=get_finance_chain_config(), display=display)
    }
        
        usage_block = f"""
        1. {chains[1].usage} [1].
        2. {chains[2].usage} [2].
        3. {chains[3].usage} [3].
        4. {chains[4].usage} [4].
        """
        template = """
            Consider the following problem : {question}. Based on the characteristics of the problem,
            identify the most suitable knowledge domain to query from the items provided. Consider each carefully 
            in the context of the question, write out the likelihood of relevant of each, and then select the most 
            appropriate knowledge domain:""" + usage_block + """
                Based on the characteristics of the given problem , select the domain that aligns most closely with the nature of the problem. It is important to first provide the number of the technique that best solves the problem, followed by a period. Then you may provide your reason why you have chosen this technique. 

            The number and name of the selected strategy is...
            """
        api_key = settings.get_api_key()
        super().__init__(api_key=api_key, template = template, destination_chains=chains, usage=config.usage, llm=config.llm_class, question=question)
        self.api_key = settings.get_api_key()
        print("Creating Knowledge Domain Router with config: ")
        # pprint.pprint(config)
        self.llm = config.llm_class(temperature=config.temperature, max_tokens=config.max_tokens, api_key=self.api_key)
        self.question: str = question

    def run(self, question: str) -> str:


        for index, chain in self.destination_chains.items():
            print(f'Running Chain #{index}')
            self.display(f"Running Chain #{index}")
            response = chain.run(question)
            print(response)
            self.display(response)
        return response


    @staticmethod
    def find_first_integer(text: str) -> Optional[int]:
        match = re.search(r'\d+', text)
        if match:
            return int(match.group())
        else:
            return None

    def determine_and_execute(self, question) -> Tuple[str, str]:
        prompt = PromptTemplate(template=self.template, input_variables=["question"])
        llm_chain = LLMChain(prompt=prompt, llm=self.llm)

        response = llm_chain.run(self.question)
        print(response)
        self.display(response)
        n = self.find_first_integer(response)

        if n in self.destination_chains:
            chain_resp = self.destination_chains[n].run(self.question)
        else:
            chain_resp = (f"Chain number {n} is not recognized.")
            print(chain_resp)
            
        return response, chain_resp

def get_knowledge_domain_router_config(temperature: float = 0.6) -> LLMChainConfig:
    usage="This router should be used when determining the most effective strategy for a query requiring domain-specific knowledge to derive"
    return LLMChainConfig(temperature=temperature, max_tokens=3000, usage=usage)
