import os
from langchain.agents import AgentType
from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI
from tools.issue_search import JiraSearchTool

model = ChatOpenAI(model="gpt-3.5-turbo-0613")
tools = [JiraSearchTool()]
open_ai_agent = initialize_agent(tools,
                                 model,
                                 agent=AgentType.OPENAI_FUNCTIONS,
                                 verbose=True)

while True:
    try:
        question = input("Question: ")
        tool_result = open_ai_agent.run(question)
        print("Answer: ", tool_result)
    except KeyboardInterrupt:
        break
    