import os
from langchain.chat_models import ChatOpenAI
from prompts.auto_generator import (
    AutoMarketAnalysisPromptGenerator, 
    AutoCompetitorAssessmentPromptGenerator,
    AutoDetectUniqueSellingPointPromptGenerator, 
    AutoContentCreationPromptGenerator
)
from agent.agent import Agent
from viewmodel.model import CompanyInfo
from typing import Union, Tuple, Any
from utils.prompt_utils import add_language
from agent.tools import search_company_db

from apikey import load_env
OPENAI_API_KEY, SERPER_API_KEY = load_env()
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

class LLMDynamicChat:
    def __init__(
            self,
            llm : ChatOpenAI | None = None,
            agent: Agent | None = None,
            market_analysis_prompt_generator : AutoMarketAnalysisPromptGenerator | None = None,
            competitor_analysis_prompt_generator : AutoCompetitorAssessmentPromptGenerator | None = None,
            selling_point_analysis_prompt_generator : AutoDetectUniqueSellingPointPromptGenerator | None = None,
            content_creation_prompt_generator : AutoContentCreationPromptGenerator | None = None,
            temperature : float = 0.0,
            language : str = "English"
    ):
        if llm is None:
            llm = ChatOpenAI(
                model_name="gpt-3.5-turbo", 
                temperature=temperature, 
                openai_api_key=OPENAI_API_KEY
            )
        self.llm = llm
        if market_analysis_prompt_generator is None:
            market_analysis_prompt_generator = AutoMarketAnalysisPromptGenerator(llm=self.llm)
        if competitor_analysis_prompt_generator is None:
            competitor_analysis_prompt_generator = AutoCompetitorAssessmentPromptGenerator(llm=self.llm)
        if selling_point_analysis_prompt_generator is None:
            selling_point_analysis_prompt_generator = AutoDetectUniqueSellingPointPromptGenerator(llm=self.llm)
        if content_creation_prompt_generator is None:
            content_creation_prompt_generator = AutoContentCreationPromptGenerator(llm=self.llm)

        if agent is None:
            agent = Agent(
                llm = llm,
                tools=["google-serper"],
            )
        
        self.llm = llm
        self.market_analysis_prompt_generator = market_analysis_prompt_generator
        self.competitor_analysis_prompt_generator = competitor_analysis_prompt_generator
        self.selling_point_analysis_prompt_generator = selling_point_analysis_prompt_generator
        self.content_creation_prompt_generator = content_creation_prompt_generator
        self.agent = agent
        self.language = language
    def auto_analysis_company(
            self,
            companyInfo : CompanyInfo,
            returning_memory : bool = True
    ) -> Union[dict, Tuple[dict, Any]]: 
        
        market_analysis_prompt = self.market_analysis_prompt_generator.generate_dynamic_prompt(domain=companyInfo.domain)
        # market_analysis_prompt = add_language(
        #         language=self.language, 
        #         base_prompt=market_analysis_prompt
        #     )
        market_analysis = self.agent.answer(market_analysis_prompt)
        
        competitor_prompt = self.competitor_analysis_prompt_generator.generate_dynamic_prompt(domain=companyInfo.domain)
        # competitor_prompt = add_language(
        #         language=self.language, 
        #         base_prompt=competitor_prompt
        #     )
        competitor_analysis = self.agent.answer(competitor_prompt)

        selling_point_prompt = self.selling_point_analysis_prompt_generator.generate_dynamic_prompt(domain=companyInfo.domain, competitor_analysis=competitor_analysis)
        # selling_point_prompt = add_language(
        #         language=self.language, 
        #         base_prompt=selling_point_prompt
        #     )
        selling_point_analysis, memory= self.agent.answer(selling_point_prompt, returning_memory=returning_memory)

        companyResearchInfo = {
            "market_analysis": market_analysis,
            "competitor": competitor_analysis,
            "key_selling_point": selling_point_analysis
        }

        return companyResearchInfo, memory
        
            