from langchain.chat_models import ChatOpenAI

from langchain.llms import OpenAI
from langchain.agents import load_tools, initialize_agent

from langchain.agents import AgentType
import dotenv
import openai
import os

dotenv.load_dotenv()

openai.api_key = os.environ.get("OPENAI_API_KEY")


llm = ChatOpenAI(temperature=0.5)
math_llm = OpenAI(temperature=0.5)

tools = load_tools(["human", "llm-math"],
                   llm=math_llm)

agent_chain = initialize_agent(
    tools, 
    llm, 
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

agent_chain.run("What is my math problem and its solution?")