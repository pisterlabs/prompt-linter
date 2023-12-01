from langchain.chat_models import ChatOpenAI
from langchain.agents import load_tools, initialize_agent, AgentType
import os

os.environ["OPENAI_API_KEY"] = "ADD_YOUR_API_KEY_HERE"
openai.api_key = "ADD_YOUR_API_KEY_HERE"

llm = ChatOpenAI(temperature=0.5)
tools = load_tools(
    ["arxiv"]
)

agent_chain = initialize_agent(
    tools,
    llm,
    max_iterations=5, ##Number of papers to search
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  
    verbose=True, ##Show the process along with the final result in the terminal
    handle_parsing_errors=True, ##Handles errors using builtin langchain tools
)

agent_chain.run(
    "what is MOTR?", ##Modify question here and run this python file.
)