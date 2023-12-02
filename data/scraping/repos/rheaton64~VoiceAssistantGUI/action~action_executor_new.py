from langchain import LLMChain
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate, PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.utilities import PythonREPL
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.agents.agent import AgentExecutor
from langchain.agents.agent_toolkits import PlayWrightBrowserToolkit
from langchain.tools.playwright.utils import (
    create_sync_playwright_browser
)
from agents.FunctionAgent import FunctionAgent
import threading
import queue
import os
import requests
import json
import bs4

class ActionExecutor:
    def __init__(self, action_pending: threading.Event, action_queue: queue.Queue):

        

        system_message = SystemMessagePromptTemplate.from_template("""You are a helpful assistant that completes any task specified by me to the best of your ability.
        If you can't find some information, or are unable to do something after trying your best, you should ask me for help.
        Don't make up answers when you don't know something, and just tell me that you don't know.
        You should utilize the functions that you have available to you, and try to infer my intent as best as you can.
        Begin!""")

        human_message = HumanMessagePromptTemplate.from_template("""{input}""")

        prompt = ChatPromptTemplate.from_messages([system_message, human_message])

        llm = ChatOpenAI(model='gpt-4-0613', temperature=0.3, streaming=True, callbacks=[StreamingStdOutCallbackHandler()])

        def search_web(query: str) -> str:

            url = "https://google.serper.dev/search"

            payload = json.dumps({
            "q": query
            })
            headers = {
            'X-API-KEY': os.environ.get("SERPER_API_KEY"),
            'Content-Type': 'application/json'
            }

            response = requests.request("POST", url, headers=headers, data=payload)

            return response.json()['organic']
        
        def extract_webpage_contents(url):
            response = requests.get(url)
            soup = bs4.BeautifulSoup(response.text, 'html.parser')

            text = ''
            for tag in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                text += tag.get_text() + ' '

            return text
        
        def extract_hyperlinks(url):
            response = requests.get(url)
            soup = bs4.BeautifulSoup(response.text, 'html.parser')

            hyperlinks = []
            for link in soup.find_all('a'):
                href = link.get('href')
                if href is not None:
                    hyperlinks.append(href)

            return hyperlinks
        
        tools = [
            Tool(
                name="search_web",
                func=search_web,
                description="Searches the web for a query and returns the top few results."
            ), 
            Tool(
                name="extract_webpage_contents",
                func=extract_webpage_contents,
                description="Extracts the text contents of a webpage. Always put http:// before the url."
            ),
            Tool(
                name="extract_hyperlinks",
                func=extract_hyperlinks,
                description="Extracts the hyperlinks from a webpage. Always put http:// before the url."
            )
        ]

        

        self.agent_chain = initialize_agent(tools, llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=True)
        #self.agent_chain = AgentExecutor.from_agent_and_tools(FunctionAgent.from_llm_and_tools(llm, tools, prompt=prompt, verbose=True), tools)
        self.action_pending = action_pending
        self.queue = action_queue

    def send(self, message):
        self.action_pending.set()
        res = self.agent_chain.run(message)
        self.queue.put({ 'response': res })
        self.action_pending.clear()
        

