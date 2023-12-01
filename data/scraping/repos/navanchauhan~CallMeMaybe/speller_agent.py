import logging
from typing import Optional, Tuple
import typing
from vocode.streaming.agent.chat_gpt_agent import ChatGPTAgent
from vocode.streaming.models.agent import AgentConfig, AgentType, ChatGPTAgentConfig
from vocode.streaming.agent.base_agent import BaseAgent, RespondAgent
from vocode.streaming.agent.factory import AgentFactory

import os
import sys
import typing
from dotenv import load_dotenv

from lang_prompt_demo import tools

from langchain.memory import ConversationBufferMemory
from langchain.utilities import SerpAPIWrapper

from langchain.agents import load_tools

from stdout_filterer import RedactPhoneNumbers

load_dotenv()

from langchain.chat_models import ChatOpenAI
# from langchain.chat_models import BedrockChat
from langchain.agents import initialize_agent
from langchain.agents import AgentType as LangAgentType


llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")  # type: ignore
#llm = BedrockChat(model_id="anthropic.claude-instant-v1", model_kwargs={"temperature":0})  # type: ignore
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    # Logging of LLMChains
verbose = True
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=LangAgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=verbose,
    memory=memory,
)


class SpellerAgentConfig(AgentConfig, type="agent_speller"):
    pass


class SpellerAgent(RespondAgent[SpellerAgentConfig]):
    def __init__(self, agent_config: SpellerAgentConfig):
        super().__init__(agent_config=agent_config)

    async def respond(
        self,
        human_input,
        conversation_id: str,
        is_interrupt: bool = False,
    ) -> Tuple[Optional[str], bool]:
        print("SpellerAgent: ", human_input)
        res = agent.run(human_input)
        return res, False


class SpellerAgentFactory(AgentFactory):
    def create_agent(
        self, agent_config: AgentConfig, logger: Optional[logging.Logger] = None
    ) -> BaseAgent:
        print("Setting up agent", agent_config, agent_config.type)
        if agent_config.type == AgentType.CHAT_GPT:
            return ChatGPTAgent(
                agent_config=typing.cast(ChatGPTAgentConfig, agent_config)
            )
        elif agent_config.type == "agent_speller":
            return SpellerAgent(
                agent_config=typing.cast(SpellerAgentConfig, agent_config)
            )
        raise Exception("Invalid agent config")