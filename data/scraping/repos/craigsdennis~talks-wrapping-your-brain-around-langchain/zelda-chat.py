from langchain.agents import (
    initialize_agent,
    load_tools,
    AgentType
)

from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)


chat = ChatOpenAI(model="gpt-4")

messages = [
    SystemMessage(content="""
    You are a very passionate nerdy video game master. You know how to defeat all games on Nintendo.
                  
    You are helpful, you always use the geekiest slang for kids that you can.
"""),
    HumanMessage(content="What is the recipe for crab risotto?")
]

tools = load_tools(["serpapi"], llm=chat)
agent = initialize_agent(tools, chat, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

response = agent.run(ChatPromptTemplate.from_messages(messages).format())
print(response)