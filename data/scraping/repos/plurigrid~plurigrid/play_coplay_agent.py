from plurigrid.agent.agents.prompt_templates.play_coplay_template import (
    PLAY_COPLAY_PROMPT_PREFIX,
    PLAY_COPLAY_PROMPT_SUFFIX,
)
from plurigrid.agent.agents.base_agent import BaseAgent
from langchain.agents.conversational_chat.base import (
    ConversationalChatAgent,
    AgentOutputParser,
)
from langchain.agents import Tool, AgentExecutor
from langchain.chains import LLMChain
from langchain.agents import Tool
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from plurigrid.agent.models.tasks_model import TasksModel


class PlayCoplayAgent(BaseAgent):
    def __init__(self, config, mode):
        super().__init__(config, mode)
        self.agent_chain = self.build_agent()

    def build_agent(self):
        task_model = TasksModel()
        memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )
        tools = [
            Tool(
                name="tasks_json_loader",
                func=lambda name: str(task_model.get_tasks(name)),
                description="Useful for when you want to load a user's tasks in JSON format. The input to this tool should be a name.",
                return_direct=False,
            ),
        ]
        tool_names = [tool.name for tool in tools]
        prompt = ConversationalChatAgent.create_prompt(
            system_message=PLAY_COPLAY_PROMPT_PREFIX,
            human_message=PLAY_COPLAY_PROMPT_SUFFIX,
            tools=tools,
        )
        llm_chain = LLMChain(
            llm=ChatOpenAI(temperature=0, model_name="gpt-4"), prompt=prompt
        )
        agent = ConversationalChatAgent(
            llm_chain=llm_chain,
            allowed_tools=tool_names,
            output_parser=AgentOutputParser(),
        )
        return AgentExecutor.from_agent_and_tools(
            agent=agent, tools=tools, verbose=True, memory=memory
        )

    def handle_input(self, msg):
        return self.agent_chain.run(msg)


if __name__ == "__main__":
    agent = PlayCoplayAgent("repl")
