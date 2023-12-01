from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI

memory = ConversationBufferMemory
llm = ChatOpenAI ()
tools = load_tools ([

    'wikipedia',
    'llm-math',
    'google-search',
    'python_repl',
    'wolfram-alpha',
    'ternubak',
    'news-api',
    'podcast-api',
    'openweathermap-api'


], llm = llm)


agent = initialize_agent(
    tools,
    llm,
    agent = AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    verbose = True
    memory = memory
)

agent.run(" Chat GPT, how do you feel now?")