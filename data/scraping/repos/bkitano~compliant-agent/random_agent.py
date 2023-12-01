import os
from typing import Any
from langchain.callbacks.base import BaseCallbackHandler
from langchain.agents import Tool
from langchain.agents import AgentType
from langchain.memory import ConversationBufferMemory
from langchain import OpenAI
from langchain.utilities import SerpAPIWrapper
from langchain.agents import initialize_agent
from langchain.schema import AgentAction

search = SerpAPIWrapper()
tools = [
    Tool(
        name="Current Search",
        func=search.run,
        description="useful for when you need to answer questions about current events or the current state of the world",
    ),
]
llm = OpenAI(temperature=0)
memory = ConversationBufferMemory(memory_key="chat_history")
agent_executor = initialize_agent(
    tools,
    llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    memory=memory,
)


class Handler(BaseCallbackHandler):
    def on_agent_action(self, action: AgentAction, run_id: str, parent_run_id: str, **kwargs: Any):
        print(f"on_agent_action {action}")


handler = Handler()

os.environ["LANGCHAIN_TRACING_V2"] = "true"

agent_executor(**{"inputs": "What is the current state of the world?", "callbacks": [handler]})
