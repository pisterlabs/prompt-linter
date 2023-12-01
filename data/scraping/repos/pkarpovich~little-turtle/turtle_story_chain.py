from typing import TypedDict, List, Optional

from langchain.base_language import BaseLanguageModel
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from little_turtle.chains import ChainAnalytics
from little_turtle.prompts import TURTLE_STORY_PROMPT_TEMPLATE
from little_turtle.services import AppConfig
from little_turtle.utils import get_day_of_week


class TurtleStoryChainVariables(TypedDict):
    date: str
    comment: str
    language: str
    target_topics: List[str]


class TurtleStoryChain:
    def __init__(self, llm: BaseLanguageModel, chain_analytics: ChainAnalytics, config: AppConfig):
        self.config = config
        self.chain_analytics = chain_analytics
        self.llm_chain = LLMChain(
            prompt=PromptTemplate.from_template(TURTLE_STORY_PROMPT_TEMPLATE, template_format="jinja2"),
            llm=llm,
            output_key="story",
            verbose=config.DEBUG,
        )

    def get_chain(self) -> LLMChain:
        return self.llm_chain

    def run(self, variables: TurtleStoryChainVariables) -> str:
        return self.llm_chain.run(variables, callbacks=[self.chain_analytics.get_callback_handler])

    def enrich_run_variables(
            self,
            date: str,
            target_topics: List[str],
            generation_comment: Optional[str]
    ) -> TurtleStoryChainVariables:
        return TurtleStoryChainVariables(
            comment=generation_comment,
            target_topics=target_topics,
            date=f"{date} ({get_day_of_week(date)})",
            language=self.config.GENERATION_LANGUAGE,
        )
