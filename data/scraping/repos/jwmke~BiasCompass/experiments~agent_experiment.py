import langchain
from langchain.agents import AgentType, initialize_agent
from langchain import PromptTemplate
from langchain.agents.tools import Tool
from langchain.llms import OpenAIChat
from langchain.utilities import GoogleSerperAPIWrapper
from langchain.document_loaders import UnstructuredURLLoader
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from termcolor import colored
from dotenv import load_dotenv


# Load environment variables from the .env file
load_dotenv()

article_link = "https://www.independent.co.uk/news/world/americas/us-politics/ron-desantis-daniel-penny-jordan-neely-b2338438.html" # TODO: make configurable

response_schemas = [
    ResponseSchema(name="1", description="first news article link"),
    ResponseSchema(name="2", description="second news article link"),
    ResponseSchema(name="3", description="third news article link"),
]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()

template = """
Use the search tool to find three separate news articles are written on the same subject of this news article: {link} 
These articles must be written within the same relative time frame as the original news article.
Respond with the link of each of the three articles.
{format_instructions}
"""
initial_prompt = PromptTemplate(
    input_variables=["link"],
    template=template,
    partial_variables={"format_instructions": format_instructions}
)

llm = OpenAIChat(model_name='gpt-3.5-turbo', temperature=0.0)

serp_tool = GoogleSerperAPIWrapper(type="news")

tools = [
    Tool(name="Search Tool",
        description="Useful for searching for news articles",
        func=serp_tool.run)
]

agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, max_iterations=5)

related_article_links = output_parser.parse(agent(initial_prompt.format(link=article_link))["output"]).values()

