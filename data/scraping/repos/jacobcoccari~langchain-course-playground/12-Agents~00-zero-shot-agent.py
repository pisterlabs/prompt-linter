from langchain.agents.agent_toolkits import create_python_agent
from langchain.agents import load_tools, initialize_agent
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

tools = load_tools(["llm-math"], llm=model)

agent = initialize_agent(
    tools,
    model,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose=True,
)

agent("""What is the 25% of 300?""")
