import os
from langchain.prompts import PromptTemplate
from langchain.agents import AgentType
from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI
from tools.issue_detail import IssueDetailTool

model = ChatOpenAI(model="gpt-3.5-turbo-0613", temperature=0)
tools = [IssueDetailTool()]

template = """
You are an AI assistant specializing in analysis Jira tickets status.
You will be given a list of jira ticket key to get all detail information from those tickets.
You will help to summarized those tickets and give a conclusion.
{input}."""

_prompt = PromptTemplate(
    input_variables=["input"],
    template=template
)


prompt = _prompt.format(
    input="你是一個專案管理的PM, 而我是專案的主持人，幫我整理這張 ticket 的目前狀況如何?  幫我整理在兩百字之內，並且標注是否需要我採取任何的行動? YOUR_TICKET , Answer reply in zh-tw")

agent = initialize_agent(
    tools=tools,
    llm=model,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True
)

response = agent.run(prompt)
print(response)