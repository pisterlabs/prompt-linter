import streamlit as st

from langchain.sql_database import SQLDatabase
from langchain.llms.openai import OpenAI
from requests import get, post
from langchain.agents import (
    AgentExecutor,
    load_tools,
    initialize_agent,
    Tool,
    ZeroShotAgent,
)
from langchain.tools.python.tool import PythonREPLTool
from langchain.agents.agent_toolkits import create_python_agent
from langchain.python import PythonREPL
from langchain import SerpAPIWrapper
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from scrape_utils import scrape_websites_deep_search

from dotenv import load_dotenv


load_dotenv()

import os

open_ai_key = os.environ.get("OPENAI_API_KEY")
os.environ["SERPAPI_API_KEY"] = os.environ.get("SERPAPI_API_KEY")
memory = ConversationBufferMemory(memory_key="chat_history")


def scrape_leads(leads):
    print("leads", leads)
    tools = [PythonREPLTool()]
    prefix = """You are provided an array of URLS. Scrape these URLs for emails, phone numbers and names of people associated from the business.
      save the emails to a CSV file with columns for each type of contact being searched. You have access to the following tools:"""
    suffix = """Begin! Remember to look for emails when scraping and save the emails to a local CSV file with the columns: "Website" and "Email".

    Question: {input}
    {agent_scratchpad}"""

    prompt = ZeroShotAgent.create_prompt(
        tools,
        prefix=prefix,
        suffix=suffix,
        input_variables=["input", "agent_scratchpad"],
    )
    llm_chain = LLMChain(
        llm=OpenAI(temperature=0, openai_api_key=open_ai_key, max_tokens=1000),
        prompt=prompt,
    )
    tool_names = [tool.name for tool in tools]
    agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names)
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent, tools=tools, verbose=True
    )
    emails = agent_executor.run(leads)


def find_leads(query):
    search = SerpAPIWrapper()
    pythonRepl = PythonREPLTool()
    tools = [
        Tool(
            name="Search",
            func=search.run,
            description="useful for when you need to search the internet based on a query",
        ),
        Tool(
            name="Python REPL",
            func=pythonRepl.run,
            description="useful for when you need to ping a website for a 200 response",
        ),
    ]
    # prefix = """Find leads by searching existing website based on the type of leads asked for in the question and returning their URLs into a dictionary where each key for a URL is "url".
    # Check that the website actually return a 200 response.
    # If you see links to other pages, check those too. You have access to the following tools:"""
    prefix2 = """Search for existing websites related to the type of leads asked for in the question and return the URL links separated by commas.
    If you see links to other pages, check those too. You have access to the following tools:"""
    suffix = """Begin! Remember to return the URL links of the search results you find separated by commas.

    Question: {input}
    {agent_scratchpad}"""

    prompt = ZeroShotAgent.create_prompt(
        tools,
        prefix=prefix2,
        suffix=suffix,
        input_variables=["input", "agent_scratchpad"],
    )
    llm_chain = LLMChain(
        llm=OpenAI(temperature=0, openai_api_key=open_ai_key, max_tokens=1000),
        prompt=prompt,
    )
    tool_names = [tool.name for tool in tools]
    agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names)
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent, tools=tools, verbose=True
    )
    urls = agent_executor.run(query)
    print("urls", type(urls))
    url_list = urls.split(",")
    print("url_list", url_list)
    leads = scrape_websites_deep_search(url_list)

    # scrape_leads(url_list)
    return leads


st.header("Lead Generator")

user_input = st.text_input("Enter the kind of leads your looking for")

if "memory" not in st.session_state:
    st.session_state["memory"] = ""

if st.button("Generate leads!"):
    st.markdown(find_leads(user_input))
    # st.markdown(
    #     scrape_websites_deep_search(
    #         [
    #             # "https://www.bigcommerce.com/",
    #             " https://www.retailmenot.com/",
    #             " https://www.shipstation.com/",
    #             " https://www.gembah.com/",
    #             # " https://www.shipgate.com/",
    #             " https://www.volusion.com/",
    #             " https://www.favordelivery.com/",
    #             " https://www.cratejoy.com/",
    #             " https://www.yeti.com/",
    #             " https://syscolabs.com/",
    #         ]
    #     )
    # )

    st.session_state["memory"] += memory.buffer
    print(st.session_state["memory"])


# def find_leads_v2(query):
#     search = SerpAPIWrapper()
#     pythonRepl = PythonREPLTool()
#     tools = [
#         Tool(
#             name="Python REPL",
#             func=pythonRepl.run,
#             description="useful for when you need to scrape websites and crawl for them",
#         ),
#     ]
#     prefix = """Find leads by crawling the internet based on the type of leads asked for in the question and returning their URLs into a dictionary where each key for a URL is "url".
#     Check that the website actually return a 200 response.
#     If you see links to other pages, check those too. You have access to the following tools:"""
#     suffix = """Begin! Don't make up any URLs if you can't find any. Remember to return exact URLs of websites you find in a dictionary where the key of each URL is: "url".

#     Question: {input}
#     {agent_scratchpad}"""

#     prompt = ZeroShotAgent.create_prompt(
#         tools,
#         prefix=prefix,
#         suffix=suffix,
#         input_variables=["input", "agent_scratchpad"],
#     )
#     llm_chain = LLMChain(
#         llm=OpenAI(temperature=0, openai_api_key=open_ai_key), prompt=prompt
#     )
#     tool_names = [tool.name for tool in tools]
#     agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names)
#     agent_executor = AgentExecutor.from_agent_and_tools(
#         agent=agent, tools=tools, verbose=True
#     )
#     urls = agent_executor.run(query)
#     leads = scrape_websites_deep_search(urls)
#     return leads
