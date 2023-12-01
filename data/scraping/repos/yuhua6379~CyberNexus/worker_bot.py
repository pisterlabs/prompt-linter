from typing import List

from langchain.agents import BaseSingleActionAgent, OpenAIFunctionsAgent
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage
from langchain.tools import BaseTool

from bot.base_bot import BaseBot
from bot.body.body import Body
from model.base_prompt_factory import BasePromptFactory
from model.llm import BaseLLM
from repo.character import Character


class WorkerBot(BaseBot):
    """这个机器人有自己的四肢，可以调用工具做事情"""

    def __init__(self, llm: BaseLLM,
                 character: Character,
                 factory: BasePromptFactory,
                 tools: List[BaseTool],
                 agent: BaseSingleActionAgent = None
                 ):
        super().__init__(llm, character, factory)
        if agent is None:
            llm = ChatOpenAI(temperature=0)
            system_message = SystemMessage(
                content="")
            prompt = OpenAIFunctionsAgent.create_prompt(system_message=system_message)
            agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt)
        self.body = Body(character, agent, factory, tools)

    def do_something(self, something_to_do):
        return self.body.do_something(something_to_do)
