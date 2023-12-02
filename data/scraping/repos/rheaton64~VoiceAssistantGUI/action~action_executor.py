from langchain import LLMChain
from langchain.schema import SystemMessage
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate, PromptTemplate
from langchain.tools.python.tool import PythonREPLTool
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, validator
from langchain.utilities import PythonREPL
import threading
import queue
import os
import requests
import json
import bs4


class ActionResponse(BaseModel):
    explain: str = Field(description="What is the user asking you to do? What outputs are the user expecting? What inputs are the user specifying, and how can we infer potential general function parameters from them? Are there any ambiguities in the task? What are they? How can they be accounted for? Can we solve these ambiguities in the task without asking the user for more information?")
    plan: list[str] = Field(description="How to complete the task step by step. Be sure to mention any predefined helper variables that you will use, such as API keys.")
    libraries: list[str] = Field(description="What non-standard libraries, if any, need to be installed to complete the task?")
    code: str = Field(description="The Python code to complete the task. All libraries are already imported in the REPL. Should consist of any helper functions or variables, the new generic function, and the function call to run it.")
    notes: str = Field(description="Can this code run without any edits from the user? If not, why? Also, any additional notes regarding the code or it's execution, not including the external libraries mentioned previously.")

    def to_dict(self):
        return {
            "explain": self.explain,
            "plan": self.plan,
            "libraries": self.libraries,
            "code": self.code,
            "notes": self.notes
        }

class ActionExecutor:
    def __init__(self, callbacks: BaseCallbackHandler, action_pending: threading.Event, action_queue: queue.Queue):

        
        self.output_parser = PydanticOutputParser(pydantic_object=ActionResponse)

        SYSTEM_TEMPLATE="""You are a helpful assistant that writes a Python function to complete any task specified by me.
        The function should always be able to run as-is, without any modifications for the user. You should try your best to infer the user's intent, and account for any ambiguities in the task.
        The function that you write will be re-used in the future for building more complex functions. Therefore, you should make it generic and re-usable.
        The environment where the function will be run is a REPL on my local machine, and it has access to the Internet and the Python standard library. It also already has imported the 'json', 'requests', and 'bs4' libraries. To install any external libraries, you must specify them in your response.

        You should also be sure to provide the necessary code to execute the task, and use the generic function with the desired inputs.

        You can use the following helper variables and functions in your code, as they will be defined in the environment where the function will be run:
        - NEWS_API_KEY: An API key for the News API.
        - search_web(query: str) -> dict: A function that searches the web for the given query, and returns a list of search results. The function returns a list of dictionaries, where each dictionary has the following keys: 'title', 'link', 'snippet'.

        At each round of conversation, I will provide you with 
        Task: the task that you must complete by writing a Python function.

        The only output that I can see is what the function prints out. I cannot see any other output, or what the function returns.

        When given a task, you respond as described in your format instructions

        {format_instructions}
        """

        human_message = HumanMessagePromptTemplate.from_template("""Task: {task}""")

        system_prompt = PromptTemplate(
            template=SYSTEM_TEMPLATE,
            input_variables=[],
            partial_variables={
                "format_instructions": self.output_parser.get_format_instructions()
            }
        )

        system_message = SystemMessagePromptTemplate(prompt=system_prompt)

        self.chain = LLMChain(
            prompt = ChatPromptTemplate.from_messages([system_message, human_message]),
            llm = ChatOpenAI(model_name='gpt-4', temperature=0, streaming=True, callbacks=[StreamingStdOutCallbackHandler()]),
        )

        # Global functions for the REPL
        def search_web(query: str) -> str:
            import requests
            import json

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

        news_api_key = os.getenv("NEWS_API_KEY")

        self.repl = PythonREPL(
            _globals={
                "search_web": search_web,
                "NEWS_API_KEY": news_api_key,
                "json": json,
                "requests": requests,
                "bs4": bs4
            }
        )
        self.action_pending = action_pending
        self.queue = action_queue

    def send(self, message):
        self.action_pending.set()
        res = self.chain.run(message)
        res = self.output_parser.parse(res)
        out = self.repl.run(res.code)
        self.queue.put({'response': res.to_dict(), 'output': out})
        self.action_pending.clear()
        

