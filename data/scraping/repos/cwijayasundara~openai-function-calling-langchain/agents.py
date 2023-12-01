import os
import openai

os.environ['OPENAI_API_KEY'] = ""
os.environ['SERPAPI_API_KEY'] = ""
openai.api_key = os.environ['OPENAI_API_KEY']
llm_name = "gpt-4"

from langchain.agents import initialize_agent, AgentType, Tool
from langchain.chains import LLMMathChain
from langchain.chat_models import ChatOpenAI
from langchain.utilities import SerpAPIWrapper

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
search = SerpAPIWrapper()
llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about current events. You should ask targeted "
                    "questions"
    ),
    Tool(
        name="Calculator",
        func=llm_math_chain.run,
        description="useful for when you need to answer questions about math"
    ),

]

agent = initialize_agent(tools, llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=True)
agent.run("Who is Leo DiCaprio's girlfriend? What is her current age ?")
