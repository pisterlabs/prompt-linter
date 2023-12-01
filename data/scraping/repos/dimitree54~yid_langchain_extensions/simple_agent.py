from typing import Any, Sequence, List, Tuple

from langchain import BasePromptTemplate, LLMChain
from langchain.agents import Agent, AgentOutputParser
from langchain.base_language import BaseLanguageModel
from langchain.prompts import ChatPromptTemplate
from langchain.schema import AgentAction, BaseMessage, AIMessage, FunctionMessage
from langchain.tools import BaseTool

from yid_langchain_extensions.agent.raw_output_agent_executor import AgentWithThoughtsExecutor


class SimpleAgent(Agent):
    prompt: ChatPromptTemplate
    stop_sequences: List[str]

    @classmethod
    def from_llm_and_prompt(
            cls, llm: BaseLanguageModel, prompt: ChatPromptTemplate, output_parser: AgentOutputParser, **kwargs
    ) -> 'SimpleAgent':
        llm_chain = LLMChain(
            llm=llm,
            prompt=prompt
        )
        return cls(llm=llm, llm_chain=llm_chain, prompt=prompt, output_parser=output_parser, **kwargs)

    def _construct_scratchpad(
        self, intermediate_steps: List[Tuple[AgentAction, str]]
    ) -> List[BaseMessage]:
        """Construct the scratchpad that lets the agent continue its thought process."""
        thoughts: List[BaseMessage] = []
        for action, observation in intermediate_steps:
            thoughts.append(AIMessage(content=action.log))
            thoughts.append(FunctionMessage(name=action.tool, content=observation))
        return thoughts

    @property
    def _stop(self) -> List[str]:
        return self.stop_sequences

    @property
    def observation_prefix(self) -> str:
        return ""

    @property
    def llm_prefix(self) -> str:
        return ""

    @classmethod
    def create_prompt(cls, tools: Sequence[BaseTool]) -> BasePromptTemplate:
        raise NotImplementedError

    @classmethod
    def _get_default_output_parser(cls, **kwargs: Any) -> AgentOutputParser:
        raise NotImplementedError

    def get_executor(self, tools: List[BaseTool], **kwargs: Any) -> AgentWithThoughtsExecutor:
        return AgentWithThoughtsExecutor(agent=self, tools=tools, **kwargs)
