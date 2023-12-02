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
from langchain.agents import Tool, AgentExecutor
from langchain.chains.conversation.memory import ConversationBufferMemory


class OntologyAgent(BaseAgent):
    def __init__(self, config, mode):
        super().__init__(config, mode)
        self.index_model = IndexModel(config)
        self.index = self.index_model.get_index()

    def build_agent(self, index):
        print("building agent..")
        memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )
        tools = [
            Tool(
                name="knowledge_base",
                func=lambda q: str(index.query(q)),
                description="useful for when you want to answer questions about the knowledge base. The input to this tool should be a complete english sentence.",
                return_direct=False,
            ),
        ]
        llm = ChatOpenAI(
            streaming=True,
            temperature=0,
            model_name="gpt-4",
        )
        agent = ConversationalChatAgent.from_llm_and_tools(
            llm=llm,
            tools=tools,
            output_parser=AgentOutputParser(),
        )
        return AgentExecutor.from_agent_and_tools(
            agent=agent, tools=tools, verbose=True, memory=memory
        )

    def handle_input(self, msg):
        return self.agent_chain(msg)


if __name__ == "__main__":
    agent = OntologyAgent(config.Config(), "fastapi")
