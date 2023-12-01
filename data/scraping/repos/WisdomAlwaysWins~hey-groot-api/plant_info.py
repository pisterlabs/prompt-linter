from langchain.tools import BaseTool
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
# ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
from langchain.agents.agent_types import AgentType

from langchain_experimental.agents.agent_toolkits import create_csv_agent
class PlantInfo(BaseTool):
    name = "Plant_Info"
    description = """It's a good tool to use when asking about plant information. Summarize it in 100 characters"""

    def _run(self, query: str) -> str:
        agent = create_csv_agent(
        ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613"),
        "static/plant_info_.csv",
        verbose=False,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        )
        return agent.run(query)
    
    async def _arun(self, query: str) -> str:
        raise NotImplementedError("질문에 답할 수 없어요.")
    