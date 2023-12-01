from langchain.agents.agent_toolkits import GmailToolkit
from langchain import OpenAI
from langchain.agents import initialize_agent, AgentType

toolkit = GmailToolkit()

llm = OpenAI(temperature=0, openai_api_key="" )
agent = initialize_agent(
    tools=toolkit.get_tools(),
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
)


agent.run(
    "Create a gmail draft for me to edit of a letter from the perspective of a sentient parrot"
    " who is looking to collaborate on some research with her"
    " estranged friend, a cat. Under no circumstances may you send the message, however."
)