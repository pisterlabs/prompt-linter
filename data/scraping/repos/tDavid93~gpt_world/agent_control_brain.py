from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    BaseMessage,
)
from langchain.output_parsers import RegexParserimport


class agent_control_brain:
    def __init__(self, agent):
        self.agent = agent
        self.model = ChatOpenAI(temperature=0)

        self.instrcutions = """
            Your goal is to maximaze your health. You can do it by 
            """
