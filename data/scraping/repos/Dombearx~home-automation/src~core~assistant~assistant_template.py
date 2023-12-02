from typing import Callable, Optional

from langchain.agents import AgentExecutor
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.chat_models.base import BaseChatModel
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import Runnable
from langchain.tools import BaseTool


def identity_function(x):
    return x


class ChatBotTemplate:
    def __init__(
        self,
        main_llm: BaseChatModel,
        tools: Optional[list[BaseTool]] = None,
        format_function: Callable = identity_function,
        tool_format_function: Callable = identity_function,
    ):
        self.tools = tools
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a home assistant, your goal is to listen to human orders. In order to respond to user allways use RespondToUserTool.",
                ),
                ("user", "{human_input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        if tools:
            main_llm = self.bind_tools(
                main_llm, tools=tools, format_function=tool_format_function
            )

        self.agent = (
            {
                "human_input": lambda x: x["human_input"],
                "agent_scratchpad": lambda x: format_function(x["intermediate_steps"]),
            }
            | prompt
            | main_llm
            | OpenAIFunctionsAgentOutputParser()
        )

    def chat(self, human_input: str):
        agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True)
        output = agent_executor.invoke({"human_input": human_input})

        return output["output"]

    def bind_tools(
        self,
        llm: BaseChatModel,
        tools: Optional[list[BaseTool]],
        format_function: Callable,
    ) -> Runnable:
        return llm.bind(functions=[format_function(tool) for tool in tools])
