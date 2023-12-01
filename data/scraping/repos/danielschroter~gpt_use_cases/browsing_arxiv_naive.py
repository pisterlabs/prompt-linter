from langchain.chat_models import ChatOpenAI

from langchain.agents import load_tools, initialize_agent, AgentType

import os
import openai
import dotenv

dotenv.load_dotenv()

openai.api_key = os.environ.get("OPENAI_API_KEY")

llm = ChatOpenAI(temperature=0.3)

tools = load_tools(["arxiv", "pubmed"])

agent_chain = initialize_agent(
    tools=tools,
    llm=llm,
    max_iterations=5,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)

agent_chain.run(
    "what is RLHF?"
)
