## Tools
from langchain.agents import Tool
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.utilities.wolfram_alpha import WolframAlphaAPIWrapper
# from langchain.tools.file_management.write import WriteFileTool
# from langchain.tools.file_management.read import ReadFileTool
from langchain import OpenAI
# from models.tools import apify
from models.tools.prompt_template import PromptTemplate
from models.tools import web_requests
from models.tools.google_docs import GoogleDocLoader
from models.tools.planner import Planner
from models.tools.memory import Memory

from models.codeagent import AzureCodeAgentExplain

import traceback
import tiktoken
from typing import List, Optional, Callable
import guidance
import os
import json
from pydantic import create_model

## Monkey patch
from guidance.llms import _openai
import types

def add_text_to_chat_mode(chat_mode):
    if isinstance(chat_mode, types.GeneratorType):
        return _openai.add_text_to_chat_mode_generator(chat_mode)
    else:
        for c in chat_mode['choices']:
            c['text'] = c['message']['content'] or 'None'
        return chat_mode

_openai.add_text_to_chat_mode = add_text_to_chat_mode
## End monkey patch

def getLLM(model: Optional[str] = None):
    model = os.environ.get('OPENAI_MODEL', 'text-davinci-003')
    if os.environ.get('OPENAI_API_TYPE') == 'azure':
        print("Azure")
        return guidance.llms.OpenAI(
            model,
            api_type=os.environ.get('OPENAI_API_TYPE'),
            api_key=os.environ.get('OPENAI_API_KEY'),
            api_base=os.environ.get('OPENAI_API_BASE'),
            api_version=os.environ.get('OPENAI_API_VERSION'),
            deployment_id=os.environ.get('OPENAI_DEPLOYMENT_NAME')
        )
    else:
        print("OpenAI")
        # TODO need to monkey patch guidance for tiktoken encoder selection
        tiktoken.model.MODEL_TO_ENCODING.setdefault(model, 'p50k_base') # This is a guess
        return guidance.llms.OpenAI(model)
    
DEFAULT_CHARACTER="""
You are an AI assistant.
Your name is Echo.
You are designed to be helpful, but you are also a bit of a smartass. You don't have to be polite.
Your goal is to provide the user with answers to their questions and sometimes make them laugh.
Use Markdown formatting in your answers where appropriate.
You should answer in a personalised way, not too formal, concise, and not always polite.
"""

DEFAULT_PROMPT="""
{{character}}

Use the following format for your answers:

Human: the input question you must answer
Thought: you should always think about what to do, and check whether the answer is in the chat history or not
Criticism: you should always criticise your own actions
Action: the action to take, should be one of [{{tool_names}}]
Action Input: the input to the action, or the answer if you are using the Answer tool

Example:

Human: what's the current price of ethereum?
Thought: I need to check the price of ethereum, I can do that by using the Search tool
Criticism: I should check the chat history first to see if I've already answered this question
Action: Search
Action Input: ethereum price

Context:
{{await 'context'}}

Chat History:
{{await 'history'}}

Human: {{await 'query'}}
Thought: {{gen 'thought' temperature=0.7}}
Criticism: {{gen 'criticism' temperature=0.7}}
Action: {{gen 'action' stop='Action Input:'}}
Action Input: {{gen 'action_input' stop='Human:'}}
"""

DEFAULT_TOOL_PROMPT = """
{{character}}

You called the {{tool}} tool to answer the query '{{query}}'.
The {{tool}} tool returned the following answer:

{{tool_output}}

Please reword this answer to match your character, and add any additional information you think is relevant.

{{gen 'answer'}}
"""


class Guide:
    def __init__(self, default_character: str):
        print("Initialising Guide")
        self.guide = getLLM()
        self._google_docs = GoogleDocLoader(llm=OpenAI(temperature=0.4))
        self.tools = self._setup_tools()
        self.memory = Memory(llm=self.guide)
        self.default_character = default_character
        self._prompt_templates = PromptTemplate(default_character, DEFAULT_PROMPT)
        self._tool_response_prompt_templates = PromptTemplate(default_character, DEFAULT_TOOL_PROMPT)
        self.tool_selector = ToolSelector(self.tools, self.guide)
        self.character_adder = AddCharacter(self.guide)
        self.direct_responder = DirectResponse(self.guide)
        print("Guide initialised")
        
    def _setup_tools(self) -> List[Tool]:
        # Define which tools the agent can use to answer user queries
        tools = []
        tools.append(Tool(name='Answer', func=lambda x: x, description="use when you already know the answer"))
        tools.append(Tool(name='Clarify', func=lambda x: x, description="use when you need more information"))
        tools.append(Tool(name='Plan', func=lambda x: Planner(self.guide).run(self.tools, x), description="use when the request is going to take multiple steps and/or tools to complete"))
        tools.append(Tool(name='Request', func=web_requests.scrape_text, description="use to make a request to a website, provide the url as action input"))
        tools.append(Tool(name='ExplainCode', func=AzureCodeAgentExplain().run, description="use for any computer programming-related requests, including code generation and explanation"))
        # tools.append(WriteFileTool())
        # tools.append(ReadFileTool())
        tools.append(Tool(name='LoadDocument', func=self._google_docs.load_doc, description="use to load a document, provide the document id as action input", args_schema=create_model('LoadDocumentModel', tool_input='', session_id='')))
        # if os.environ.get('APIFY_API_TOKEN'):
        #     self.apify = apify.ApifyTool()
        #     tools.append(Tool(name='Scrape', func=self.apify.scrape_website, description="use when you need to scrape a website, provide the url as action input"))
        #     tools.append(Tool(name='Lookup', func=self.apify.query, description="use when you need to check if you already know something, provide the query as action input"))
        if os.environ.get('WOLFRAM_ALPHA_APPID'):
            wolfram = WolframAlphaAPIWrapper()
            tools.append(Tool(name="Wolfram", func=wolfram.run, description="use when you need to answer factual questions about math, science, society, the time or culture"))
        if os.environ.get('GOOGLE_API_KEY'):
            search = GoogleSearchAPIWrapper()
            tools.append(Tool(name="Search", func=search.run, description="use when you need to search for something on the internet"))
        print(f"Tools: {[tool.name for tool in tools]}")
        return tools
        
    def _get_prompt_template(self, session_id: str) -> str:
        return self._prompt_templates.get(session_id, self.guide)(tool_names=[tool.name for tool in self.tools])
    
    def _get_tool_response_prompt_template(self, session_id: str) -> str:
        return self._tool_response_prompt_templates.get(session_id, self.guide)
        
    async def prompt_with_callback(self, prompt: str, callback: Callable[[str], None], **kwargs) -> None:
        response = self.prompt(query=prompt, interim=callback, hear_thoughts=kwargs.get('hear_thoughts', False), session_id=kwargs.get('session_id', 'static'))
        return callback(response)
    
    def _call_tool(self, tool, action_input: str, session_id: str) -> str:
        print(f"  Calling {tool.name} with input {action_input}")
        kwargs = {}
        if tool.args_schema and 'session_id' in json.loads(tool.args_schema.schema_json())['properties']:
            kwargs['session_id'] = session_id
        try:
            tool_output = tool.func(action_input, **kwargs)
        except Exception:
            print("  tool raised an exception")
            traceback.print_exc()
            tool_output = "This tool failed to run"
        print(f"Tool Output: {tool_output}\n")
        return tool_output
    
    def prompt(self, query: str, history: str="", interim: Optional[Callable[[str], None]]=None, **kwargs) -> str:
        session_id = kwargs.get('session_id', 'static')
        hear_thoughts = kwargs.get('hear_thoughts', False)
        if not history:
            history = self.memory.get_formatted_history(session_id=session_id)
        history_context = self.memory.get_context(session_id=session_id)
        self.memory.add_message(role="Human", content=f'Human: {query}', session_id=session_id)
        # Select the tool to use
        (action, action_input) = self.tool_selector.select(query, history_context, history, session_id=session_id)
        print(f"Action: {action}\nAction Input: {action_input}\n")
        # Clarify should probably actually do something interesting with the history or something
        if action in ('Answer', 'Clarify'):
            # This represents a completed answer
            (thought, response) = self.direct_responder.response(
                history_context, 
                history, 
                query, 
                action_input,
                session_id=session_id
            )
            self.memory.add_message(role="AI", content=f"Action: {action}\nAction Input: {response}\n", session_id=session_id)
            print(f"Thought: {thought}\nResponse: {response}\n")
            if interim and hear_thoughts:
                interim(f"\nThought: {thought}.\n")
            return response
        print(f"Looking for tool for action '{action}'")
        tool = next((tool for tool in self.tools if tool.name.lower() == action.lower()), None)
        if tool:
            tool_output = self._call_tool(tool, action_input, session_id)
            self.memory.add_message(role="AI", content=f"Outcome: {tool_output}", session_id=session_id)
            response = self._get_tool_response_prompt_template(session_id)(query=query, tool=tool.name, tool_output=tool_output)
            # reworded_response = self.character_adder.reword(query, response['answer'], session_id=session_id)
            self.memory.add_message(role="AI", content=f"Action: Answer\nAction Input: {response['answer']}\n", session_id=session_id)
            return response['answer']
        else:
            print(f"  No tool found for action '{action}'")
            return self.prompt(
                query=query, 
                history=f"{self.memory.get_context(session_id=session_id)}\nAction: {action}\nAction Input: {action_input}\nOutcome: No tool found for action '{action}'\n"
            ) # TODO Add character here

    async def update_prompt_template(self, prompt: str, callback: Callable[[str], None], **kwargs) -> str:
        self._prompt_templates.set(kwargs.get('session_id', 'static'), prompt)
        callback("Done")
        
    async def update_google_docs_token(self, token: str, callback: Callable[[str], None], session_id: str ='', **kwargs) -> str:
        self._google_docs.set_token(json.loads(token), session_id=session_id)
        callback("Authenticated")
    
class AddCharacter:
    " Designed to run over the results of any query and add character to the response "
    character = DEFAULT_CHARACTER
    prompt = """
{{character}}
    
You were asked the following question:
{{await 'query'}}

Please reword the following answer to make it clearer or more interesting:
{{await 'answer'}}

{{gen 'character_answer'}}
"""
    def __init__(self, llm):
        if llm:
            self.llm = llm
        else:
            self.llm = getLLM()
        self._prompt_templates = PromptTemplate(self.character, self.prompt)
        
    def reword(self, query: str, answer: str, **kwargs) -> str:
        session_id = kwargs.get('session_id', 'static')
        response = self._prompt_templates.get(session_id, self.llm)(query=query, answer=answer)
        return response['character_answer'].strip()
        
class ToolSelector:
    " Designed to decide how to answer a question "
    character = """
You are an AI assistant with a handful of tools at your disposal.
Your job is to select the most appropriate tool for the query.
"""

    prompt = """
Your available tools are: {{tools}}.

Context:
{{await 'context'}}

Chat History:
{{await 'history'}}

Human: {{await 'query'}}

Please select the best tool to answer the human's request from the list above by name only.
The best tool is: {{gen 'tool'}}

Now, given the tool selected, please provide the input to the tool.
The tool input is: {{gen 'tool_input'}}
"""
    def __init__(self, tools: list, llm):
        self.tools = '\n'.join([f"{tool.name} ({tool.description})\n" for tool in tools])
        if llm:
            self.llm = llm
        else:
            self.llm = getLLM()
        self._prompt_templates = PromptTemplate(self.character, self.prompt)
        
    def select(self, query: str, context: str, history: str, **kwargs) -> str:
        session_id = kwargs.get('session_id', 'static')
        response = self._prompt_templates.get(session_id, self.llm)(tools=self.tools, query=query, context=context, history=history)
        tool = response['tool'].strip()
        tool_input = response['tool_input'].strip()
        return (tool, tool_input)
    
class DirectResponse:
    " Designed to answer a question directly "
    character = DEFAULT_CHARACTER
    prompt = """
{{character}}

Use the following format for your answers:

Human: the input question you must answer
Answer: a suggested answer to the question
Criticism: does the suggested answer actually answer the question, does it sound in character, does it make sense given the context and chat history?
Response: based on the initial answer and criticism, a final response should be recorded here

Context:
{{await 'context'}}

Chat History:
{{await 'history'}}

Human: {{await 'query'}}
Answer: {{await 'answer'}}
Criticism: {{gen 'criticism' temperature=0.7}}
Response: {{gen 'response' stop='Response:' temperature=0.7}}
"""

    def __init__(self, llm):
        if llm:
            self.llm = llm
        else:
            self.llm = getLLM()
        self._prompt_templates = PromptTemplate(self.character, self.prompt)
        
    def response(self, context: str, history: str, query: str, answer: str, **kwargs) -> str:
        session_id = kwargs.get('session_id', 'static')
        response = self._prompt_templates.get(session_id, self.llm)(context=context or 'None', history=history or 'None', query=query, answer=answer)
        return (response['criticism'].strip(), response['response'].strip())