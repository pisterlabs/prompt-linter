from langchain.agents import Tool
from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI

"""
See LangChain doc:
   https://python.langchain.com/docs/modules/agents/tools/how_to/custom_tools
   
Follow this example:
   https://blog.gopenai.com/langchain-agent-tools-for-running-functions-and-apis-dc123337eb4d
"""

def login(param):
    """provide your login logic/API call here and handle both cases i.e success and failure """
    return {"response": "logged in successfully"}


def chat(param):
    """If you're implementing a chatbot then when user will ask for any information then you can use this function to implement that logic using langchain and openai"""
    return {"response": "your response"}


def logout(param):
    """provide your logout logic/API call here and handle both cases i.e success and failure """
    print(param)
    return {"response": "logged out successfully"}


loginTool = Tool(
    name="loginTool",
    func=login,
    description="use to run the login functionality"
)

logoutTool = Tool(
    name="logoutTool",
    func=logout,
    description="use to run the logout functionality"
)

chatTool = Tool(
    name="chatTool",
    func=chat,
    description="use when you need to give information about the project. Anything irrelevant, just say I don't know."
)

tools = [
    loginTool,
    logoutTool,
    chatTool
]

llm = ChatOpenAI(temperature=0)
agent = initialize_agent(
    tools,
    llm,
    agent="zero-shot-react-description",
    verbose=True
)

print(agent.agent.llm_chain.prompt.template)
# TODO: reinitialize `agent.agent.llm_chain.prompt.template` for the desired behaviours

# ---------------
# Example run - 1
# ---------------
# Agent runs function `logout` with parameter `2ndw33d3fnn`
result = agent.run("logout my profile. my session id is 2ndw33d3fnn")
print(result)

# ---------------
# Example run - 2
# ---------------
# Agent runs function `chat`
result = agent.run("Hi copilot, dinner idea?")
print(result)
print()
