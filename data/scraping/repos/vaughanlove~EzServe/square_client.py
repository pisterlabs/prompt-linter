"""The Square Agent that accesses the square api based on prompts.
"""
from source.agent.tools import (
    OrderTool,
    FindItemIdTool,
    MakeOrderCheckoutTool,
    GetOrderTool,
)

from langchain.agents import AgentType, initialize_agent
from langchain.llms import vertexai

import re

class SquareClient():
    """SquareClient class where the LLM is instantiated.
    """
    def __init__(self, verbose=True) -> bool:
        self.llm = vertexai.VertexAI(tempurature=0)

        self.tools = [
            FindItemIdTool(),
            OrderTool(),
            MakeOrderCheckoutTool(),
            GetOrderTool(),
        ]

        assert self.llm is not None, "LLM NOT INSTANTIATED"
        assert len(self.tools) > 0, "NEED AT LEAST ONE TOOL"

        self.agent = initialize_agent(
            self.tools,
            self.llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=verbose,
        )

    def run(self, prompt):
        # this is where we can set our initial prompt.
        res = self.agent.run(prompt)
        result = re.sub(r"[^a-zA-Z0-9\s]", "", res)
        return result
    