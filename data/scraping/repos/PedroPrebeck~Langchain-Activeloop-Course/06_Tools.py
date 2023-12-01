from dotenv import load_dotenv

load_dotenv()

from langchain.llms import OpenAI
from langchain.agents import Tool
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import initialize_agent, AgentType

llm = OpenAI(model="text-davinci-003", temperature=0)

prompt = PromptTemplate(
    input_variables=["query"], template="Write a summary of the following text: {query}"
)

summarize_chain = LLMChain(llm=llm, prompt=prompt)

search = GoogleSearchAPIWrapper()

tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for finding information about recent events",
    ),
    Tool(
        name="Summarizer",
        func=summarize_chain.run,
        description="useful for summarizing texts",
    ),
]

agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

response = agent(
    "What's the latest news about the Mars rover? Then please summarize the results."
)
print(response["output"])
