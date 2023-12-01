from dotenv import load_dotenv
load_dotenv(dotenv_path='.env')
import os

from langchain.llms import OpenAI
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

llm = OpenAI(model="text-davinci-003", temperature=0)

prompt = PromptTemplate(
    input_variables=["query"],
    template="Write summary of the following text {query}?",
)

summarize_chain = LLMChain(llm=llm, prompt=prompt)

search = GoogleSearchAPIWrapper()

tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for searching information on recent events on the web",
    ),
    Tool(
        name="Summarize",
        func=summarize_chain.run,
        description="useful for summarizing text",
    )
]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

response = agent("What's the latest news about the Mars rover? Then please summarize the results.")
print(response['output'])