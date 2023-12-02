from langchain import LLMChain
from agent.agents.base_agent import BaseAgent
from langchain.agents import Tool
from agent.config import config
from agent.models.index_model import IndexModel
from langchain.chat_models import ChatOpenAI
from langchain.agents.conversational_chat.base import (
    ConversationalChatAgent,
    AgentOutputParser,
)
from langchain.agents import AgentExecutor
from langchain.chains.conversation.memory import ConversationBufferMemory


class DigitalTwin(BaseAgent):
    def __init__(self, config, mode, prompt=None):
        super().__init__(config, mode)
        self.agent_chain = self.build_agent(prompt)

    def build_agent(self, prompt):
        print("building agent..")
        memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )
        llm = ChatOpenAI(temperature=0, model_name="gpt-4")
        tools = []
        kwargs = {}
        if prompt is not None:
            kwargs["system_message"] = prompt
        agent = ConversationalChatAgent.from_llm_and_tools(
            llm=llm, tools=tools, output_parser=AgentOutputParser(), **kwargs
        )
        return AgentExecutor.from_agent_and_tools(
            agent=agent, tools=tools, verbose=True, memory=memory
        )

    def handle_input(self, msg):
        return self.agent_chain.run(msg)


if __name__ == "__main__":
    agent = DigitalTwin(config.Config(), "repl")
