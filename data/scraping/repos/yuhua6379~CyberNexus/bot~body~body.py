from typing import Sequence

from langchain.agents import BaseSingleActionAgent, AgentExecutor
from langchain.tools import BaseTool

from model.base_prompt_factory import BasePromptFactory
from model.prompt_broker import PromptBroker
from repo.character import Character


class Body:
    def __init__(self, character: Character, agent: BaseSingleActionAgent, factory: BasePromptFactory,
                 tools: Sequence[BaseTool]):
        self.agent = agent
        self.tools = tools
        self.character = character
        self.broker = PromptBroker(factory)

    def do_something(self, something_to_do: str):
        session = self.broker.do_something_prompt(self.character, something_to_do)
        agent_executor = AgentExecutor(agent=self.agent, tools=self.tools)
        return agent_executor.run(session.prompt)
