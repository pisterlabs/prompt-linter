import os
from apikey import load_env
OPENAI_API_KEY, SERPER_API_KEY = load_env()
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["SERPER_API_KEY"] = SERPER_API_KEY

from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.utilities import GoogleSerperAPIWrapper
from langchain.agents import initialize_agent, AgentType, load_tools, Tool
from prompts.analysis_template import StaticPromptTemplate
from utils.json_utils import validate_list_schema, validate_market_analysis_schema
from typing import Callable, Union, Any, Tuple
from utils.prompt_utils import add_language
from agent.agent import Agent
from agent.agent_search import SearchAgent

class LLMStaticChat:
    def __init__(
            self,
            llm : Union[ChatOpenAI, OpenAI, None] = None,
            temperature : int = 0,
            language : str = "English"
    ):
        self.temperature = temperature
        self.language = language
        if llm is None:
            llm = ChatOpenAI(
                model='gpt-3.5-turbo',
                temperature=self.temperature, 
                openai_api_key=OPENAI_API_KEY,
                verbose=True
            )
        self.llm = llm
        
        self.agent = Agent(
                llm = self.llm,
                tools=["google-serper"],
            )

        #self.agent = SearchAgent(llm = llm, temperature=0.0)
    
    def company_analysis(
            self,
            company : str,
            domain: str,
    ) -> dict: 
        prompt_template = StaticPromptTemplate()
        company_analysis = dict()
        company_analysis["market_analysis"] = self.run_agent(
            prompt=prompt_template.get_market_analysis_prompt(company, domain)
        )
        company_analysis["competitor"] = self.run_agent(
            prompt=prompt_template.get_competitor_prompt(company)
        )
        company_analysis["key_selling_point"] = self.run_agent(
           prompt=prompt_template.get_key_selling_point(company)
        )

        return company_analysis

    def run_agent_with_validation(
            self,
            prompt : str,
            validator : Callable[[str], Tuple[bool, Any]],
            max_retries : int = 3
    ) -> Union[None, Any]:
        while max_retries > 0:
            response = self.agent.answer(prompt)
            val, res = validator(response)
            if val == False:
                max_retries-=1
                continue
            else:
                return res
        return None
    
    def run_agent(
            self,
            prompt : str
    ) -> str:
        response = self.agent.answer(prompt)
        return response