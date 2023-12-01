import os
import openai
from langchain.agents import Tool
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.agents import initialize_agent
from langchain.chat_models import AzureChatOpenAI
from langchain.agents import AgentType
from langchain.agents.agent_toolkits import ZapierToolkit
from langchain.utilities.zapier import ZapierNLAWrapper

load_dotenv()

azchat=AzureChatOpenAI(
    client=None,
    openai_api_base=str(os.getenv("OPENAI_API_BASE")),
    openai_api_version="2023-03-15-preview",
    deployment_name=str(os.getenv("CHAT_DEPLOYMENT_NAME")),
    openai_api_key=str(os.getenv("OPENAI_API_KEY")),
    # openai_api_type = "azure"
)

class DocsInput(BaseModel):
    question: str = Field()

zapier = ZapierNLAWrapper()
zapier_toolkit = ZapierToolkit.from_zapier_nla_wrapper(zapier)
agent = initialize_agent(zapier_toolkit.get_tools(), azchat, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

def zapierAgent(input):
    response = agent.run(input)
    return response

def aZapierAgent(input):
    response = agent.arun(input)
    return response

def ZapierTool():
    tools = []
    tools.append(Tool(
        name = "Workflow Agent",
        func=zapierAgent,
        coroutine=aZapierAgent,
        description=f"Useful for when you need to run workflow actions like find contacts and send email. Input should be a question in complete sentence. Output will be the action result and you can use it as Final Answer.",
        args_schema=DocsInput
    ))
    return tools

def zapier():
    tools = []
    tools.extend(ZapierTool())
    return tools