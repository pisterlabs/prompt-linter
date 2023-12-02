from typing import TypedDict, List

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.language_model import BaseLanguageModel

from little_turtle.services import AppConfig

DAY_EVENT_PICKER_PROMPT = """
You are a Content Curator for a book. Your task is to select the event of the day from a provided list of events. Please follow these steps:

Step 1:
Remove the events from the list based on the following criteria:
- Events that related to politics, that has impact only on a specific country
- Events that connected to religion or religious holidays
- Negative events for human history, such as wars, disasters and tragedies
- Events that are not visually engaging

Step 2:
Sort events by following criteria:
- Events that easy to visualize should be first
- Most engaging events should be at the top of the list
- Events that influence to the human history should be at the top of the list

Step 3:
Translate events into {language}. Avoid adding any additional information to the event information

All events list: {{events|join(', ')}}

The result should be a list of five events, each separated by two new lines. Refrain from displaying any intermediate information
"""


class HistoricalEventsChainVariables(TypedDict):
    events: List[str]
    language: str


class HistoricalEventsChain:
    def __init__(self, llm: BaseLanguageModel, config: AppConfig):
        self.config = config
        prompt = ChatPromptTemplate.from_template(DAY_EVENT_PICKER_PROMPT, template_format="jinja2")

        self.chain = prompt | llm | StrOutputParser()

        self.llm_chain = LLMChain(
            prompt=PromptTemplate.from_template(DAY_EVENT_PICKER_PROMPT, template_format="jinja2"),
            llm=llm,
            verbose=config.DEBUG
        )

    def run(self, variables: HistoricalEventsChainVariables) -> str:
        return self.chain.invoke(variables)

    def enrich_run_variables(self, events: List[str]) -> HistoricalEventsChainVariables:
        return HistoricalEventsChainVariables(
            language=self.config.GENERATION_LANGUAGE,
            events=[event for event in events],
        )
