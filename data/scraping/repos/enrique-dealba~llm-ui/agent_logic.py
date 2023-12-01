import argparse
import config
import json
import logging
import os
import random
import re

from typing import List, Dict, Any, Optional, Union
from apikeys import open_ai_key, hf_key, serpapi_key, langsmith_key
from prompts import server_template, agent_template, template_with_history, template_with_history_json, template_with_history_api
from agent_utils import get_agent_thoughts, extract_code_from_response
from test_files.responses import responses_1, responses_2, code_1, code_2, code_3, code_4

from agent_utils import save_code_to_file, code_equivalence
from agent_utils import extract_function_name, python_to_text
from agent_utils import extract_code_from_response_convo
from tools.custom_tools import ask_user_tool, get_current_time_tool
from objective_utils import get_objective, save_json, extract_json
from agent_files.objectives import CatalogMaintenanceObjective, PeriodicRevisitObjective
from agent_files.objectives import SearchObjective, ScheduleFillerObjective, QualityWindowObjective
from agent_files.objectives import DATA_MODES, MARKINGS, DEFAULT_DATA_MODE, DEFAULT_MARKINGS

from flask import Flask, request, jsonify, render_template
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain import PromptTemplate, LLMChain
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.llms import OpenAI, HuggingFacePipeline
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.document_loaders import TextLoader
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from InstructorEmbedding import INSTRUCTOR

## AGENT IMPORTS ##
from langchain.agents import load_tools, initialize_agent
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.tools import BaseTool
from langchain.schema import AgentAction, AgentFinish
from langchain.prompts import StringPromptTemplate
from langchain.agents.agent_toolkits import JsonToolkit
from langchain.tools.json.tool import JsonSpec
from langchain.agents.agent_toolkits.json.prompt import JSON_PREFIX, JSON_SUFFIX
from langchain.agents import create_json_agent
from langchain.agents import AgentType
from langchain.schema.output_parser import OutputParserException
from langchain.output_parsers import PydanticOutputParser

## SKYFIELD API AGENT ##
from tools.custom_tools import get_skyfield_planets_tool, get_planet_distance_tool, get_latitude_longitude_tool
from tools.custom_tools import get_skyfield_satellites_tool, get_next_visible_time_for_satellite_tool

## VLLM IMPORTS ##
# from langchain.llms import VLLM
from prompts import api_template_with_history_mistral_1, api_template_with_history_mistral_2


os.environ['OPENAI_API_KEY'] = open_ai_key
os.environ['HUGGINGFACEHUB_API_TOKEN'] = hf_key
os.environ['SERPAPI_API_KEY'] = serpapi_key
## LangChain LangSmith Project
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = langsmith_key
os.environ['LANGCHAIN_PROJECT'] = 'edealba-llm-ui'

app = Flask(__name__)

# logging.basicConfig(level=logging.DEBUG)

class ListHandler(logging.Handler): # Inherits from logging.Handler
    def __init__(self):
        super().__init__()
        self.log = []

    def emit(self, record) -> None:
        """Keeps track of verbose outputs from agent LLM in logs."""
        self.log.append(self.format(record))

class CustomPromptTemplate(StringPromptTemplate):
    template: str
    # The list of available tools
    tools: List[Tool]
    agent_thoughts: list
    
    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Formats them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            self.agent_thoughts += [str(action.log) + "\nObservation: " + str(observation)]
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        
        kwargs["agent_scratchpad"] = thoughts
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)

class CustomOutputParser(AgentOutputParser):
    
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Checks if agent should finish
        pre_final = "Final Answer:" # deleted the ':'
        post_final = "Final Answer"
        if pre_final in llm_output:
            return AgentFinish(
                return_values={"output": llm_output.split(pre_final)[-1].strip()},
                log=llm_output,
            )
        elif post_final in llm_output:
            return AgentFinish(
                return_values={"output": llm_output.split(post_final)[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)

        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)

# handler = ListHandler()
# logging.getLogger().addHandler(handler)

class AgentServer:
    """LLM Agent Flask server with Agent operations."""

    def __init__(self, llm_mode: str='davinci', agent_mode: str='text-code',
                 template: str="", is_flask_app: bool=True):
        if is_flask_app:
            self.app = app
        else:
            self.app = None
        
        self.tokenizer = None
        self.model = None
        self.llm = None
        self.embedding = None
        self.memory = None
        self.qa_chain = None

        ## Agent LLM
        self.agent_mode = agent_mode
        self.tools = None
        self.output_parser = CustomOutputParser()
        self.max_iterations = 8
        self.agent = None
        self.agent_thoughts = []
        self.agent_history = None
        self.agent_executor = None
        self.func_config = None
        self.id = random.randint(1e5, 1e6-1) # agent id

        ## Objectives LLM (json-agent)
        self.objective = None
        self.custom_json_history = None
        self.json_agent_executor = None
        self.prev_response = None

        ## Open-Source LLMs
        self.mistral = False

        # self.template = template
        self.persist_directory = 'docs/chroma/'
        self.initialize_model(llm_mode)

    def get_tools(self, use_code_tools: bool, use_default_tools: bool,
                  use_custom_tools: bool):
        """Setup of tools for agent LLM."""
        assert self.llm is not None
        tools = []

        if use_code_tools:
            base_tools = load_tools(["python_repl",
                                     "terminal"])
        elif use_default_tools:
            base_tools = load_tools(["wikipedia",
                                     "serpapi",
                                     "python_repl",
                                     "terminal"],
                                     llm=self.llm)
            
        for tool in base_tools:
            new_tool = Tool(
                name = str(tool.name),
                func=tool.run,
                description=tool.description
                )
            if new_tool not in tools:
                tools += [new_tool]

        if use_custom_tools:
            tools += [ask_user_tool]

        return tools
            
    def init_agent(self):
        """Initializes LLM agent."""
        if self.tools is None:
            self.tools = self.get_tools(use_code_tools=True,
                                        use_default_tools=False,
                                        use_custom_tools=True)
            
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            max_iterations=self.max_iterations,
            verbose=True)
        
        if self.tools is None or self.agent is None:
            return jsonify({'error': 'Agent or tools not initialized'}), 400
        
    def init_convo_agent(self):
        """Initializes LLM agent for conversational mode (input schema)."""
        if self.tools is None:
            self.tools = self.get_tools(use_code_tools=True,
                                        use_default_tools=False,
                                        use_custom_tools=True)
        
        self.agent_history = CustomPromptTemplate(
            template=template_with_history,
            tools=self.tools,
            agent_thoughts=self.agent_thoughts,
            input_variables=["input", "intermediate_steps", "history"]
            )
        llm_chain = LLMChain(llm=self.llm, prompt=self.agent_history)
        tool_names = [tool.name for tool in self.tools]

        self.agent = LLMSingleActionAgent(
            llm_chain=llm_chain,
            output_parser=self.output_parser,
            stop=["\nObservation:"],
            allowed_tools=tool_names
            )
        
        num_memories = 3 # can play around with this (maybe 2, 3, 4, 5, etc)
        self.memory = ConversationBufferWindowMemory(k=num_memories)
        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            memory=self.memory,
            max_iterations=self.max_iterations
            )
        
        if self.agent_executor is None or self.agent is None:
            return jsonify({'error': 'Agent or AgentExecutor not initialized'}), 400
    
    def init_custom_json_agent(self, use_custom_tools: bool = True):
        json_spec = JsonSpec(dict_=self.objective.data, max_value_length=4000)
        json_toolkit = JsonToolkit(spec=json_spec)
        tools = json_toolkit.get_tools()
        custom_tools = []
        for tool in tools:
            new_tool = Tool(
                name = str(tool.name),
                func=tool.run,
                description=tool.description
            )
            custom_tools += [new_tool]

        if use_custom_tools:
            custom_tools += [get_current_time_tool]

        # TODO: template_with_history might need to be tuned/optimized
        self.custom_json_history = CustomPromptTemplate(
            template=template_with_history_json,
            tools=custom_tools,
            prefix=JSON_PREFIX,
            suffix=JSON_SUFFIX,
            agent_thoughts=self.agent_thoughts,
            input_variables=["input", "intermediate_steps", "history"]
        )

        llm_chain = LLMChain(llm=self.llm, prompt=self.custom_json_history)
        tool_names = [t.name for t in custom_tools]

        self.agent = LLMSingleActionAgent(
            llm_chain=llm_chain,
            output_parser=self.output_parser,
            stop=["\nObservation:"],
            allowed_tools=tool_names
        )

        num_memories = 3 # can play around with this (maybe 3, 4, 5, etc)
        self.memory = ConversationBufferWindowMemory(k=num_memories)
        self.tools = custom_tools

        max_iterations = 8
        self.json_agent_executor = AgentExecutor.from_agent_and_tools(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            memory=self.memory,
            max_iterations=max_iterations
        )
    
    def init_json_agent(self):
        """Initializes JSON agent for outputting JSON objects based on schemas."""
        json_spec = JsonSpec(dict_=self.objective.data, max_value_length=4000)
        json_toolkit = JsonToolkit(spec=json_spec)

        self.json_agent_executor = create_json_agent(
            llm=self.llm,
            toolkit=json_toolkit,
            verbose=True
            )
        
    def initialize_model(self, llm_mode: str):
        """Initialize LLM either locally or from OpenAI (text-davinci-003)."""
        if llm_mode == 'local':
            directory_path = config.MODEL_DIRECTORY_PATH
            self.tokenizer = AutoTokenizer.from_pretrained(directory_path)
            self.model = AutoModelForCausalLM.from_pretrained(directory_path)
        if llm_mode == 'vllm':
            self.mistral = True
            self.llm = VLLM(model=config.VLLM_MODEL,
                            trust_remote_code=True,
                            max_new_tokens=512,
                            top_k=10,
                            top_p=0.95,
                            temperature=config.TEMPERATURE)
        elif llm_mode == 'chatgpt':
            self.llm = ChatOpenAI(model_name="gpt-3.5-turbo",
                                  temperature=config.TEMPERATURE)
        elif llm_mode == 'custom_gpt':
            # 10 2 task examples: "ft:gpt-3.5-turbo-0613:personal::7r74d09R"
            # 50 5 task examples: ft:gpt-3.5-turbo-0613:personal::7seuSsRY
            self.llm = ChatOpenAI(model_name="ft:gpt-3.5-turbo-0613:personal::7seuSsRY",
                                  temperature=config.TEMPERATURE)
        elif llm_mode == 'davinci':
            self.llm = OpenAI(temperature=config.TEMPERATURE) # (text-davinci-003)
        elif llm_mode == 'gpt4':
            self.llm = ChatOpenAI(model_name="gpt-4",
                                  temperature=config.TEMPERATURE)
        
        if self.model:
            self.llm = self.initialize_local_model()
            logging.debug(f"LLM running on {self.model.device}")
        
    def initialize_local_model(self) -> HuggingFacePipeline:
        """Initialize local LLM."""
        local_pipe = pipeline("text-generation",
                               model=self.model,
                               tokenizer=self.tokenizer,
                               max_length=500)
        return HuggingFacePipeline(pipeline=local_pipe)
    
    def get_response_list(self, filepath):
        try:
            with open(filepath, 'r') as file:
                lines = [line.strip() for line in file]
            return lines
        
        except FileNotFoundError:
            print("File not found. Please check the file path.")
            return []
    
    def test_agent(self, extract_fn):
        ## TODO: Clean this up to be neater.
        responses_3 = self.get_response_list("agent_responses/responses-16036891.txt")
        responses_4 = self.get_response_list("agent_responses/responses-14552350.txt")
        responses = [responses_1, responses_2, responses_3, responses_4]
        codes = [code_1, code_2, code_3, code_4]

        assert len(responses) == len(codes)
        num_tests = len(responses)
        for i in range(num_tests):
            print(f"Running test: {i+1}")
            response = responses[i]
            code = codes[i]
            code_block = extract_fn(response)
            if not code_equivalence(code_block, code):
                print(f"Got following code: {code_block}")
                print(f"But, expected code: {code}")
                print(f"Expected {len(code)} chars but got {len(code_block)} instead.")
                return False
            assert code_equivalence(code_block, code)

        return True
    
    def save_response(self, responses):
        id = random.randint(1e7, 1e8-1)
        directory = "agent_responses"
        filename = "responses-" + str(id) + ".txt"

        if not os.path.exists(directory):
            os.makedirs(directory)
        
        with open(os.path.join(directory, filename), 'w') as f:
            for response in responses:
                f.write("%s\n" % response)

    def run_agent_tests(self):
        if self.test_agent(extract_fn=extract_code_from_response):
            print()
            print("="*50)
            print("SUCCESS: Agent tests passed!")

    def validate_convo_agent(self):
        assert self.agent_executor is not None
        assert self.memory is not None
        assert self.agent_history is not None
        assert self.agent is not None
        assert self.tools is not None

    def get_function_config(self, prompt: str,
                            py_file: str,
                            agent_dir: str = 'agent_funcs'):
        if not os.path.exists(agent_dir) or not os.path.isdir(agent_dir):
            raise FileNotFoundError(f"The directory {agent_dir} does not exist.")
        assert os.path.exists(agent_dir)
        file_path = agent_dir + "/" + py_file + ".py"
        assert os.path.isfile(file_path)

        task_func = python_to_text(file_path) # converts .py to str
        func_name = extract_function_name(task_func)

        # TODO: Avoid this kind of hardcoding
        func_schema = """{
        "value1": int OR str,
        "value2": int OR str,
        "value3": int OR str
        }
        """

        return {
            'prompt': prompt,
            'py_file': py_file,
            'agent_dir': agent_dir,
            'task_func': task_func,
            'func_name': func_name,
            'func_schema': func_schema
            }
    
    def reset_prompt(self, prompt):
        self.func_config['prompt'] = prompt

    def reset_thoughts(self):
        self.agent_thoughts = []

    def get_agent_task(self, prompt: str) -> str:
        if self.func_config is None:
            self.func_config = self.get_function_config(prompt,
                                                        py_file='task_func',
                                                        agent_dir='agent_funcs'
                                                        )
        assert self.func_config is not None
        
        if self.func_config['prompt'] is not prompt:
            # This means a new user prompt has been sent
            print()
            print("&"*50)
            print("&"*50)
            print("RESETTING PROMPT and THOUGHTS")
            self.reset_prompt(prompt)
            self.reset_thoughts()

        agent_task = f"""Given the following Python function:

        {self.func_config['task_func']}

        And using the following input schema for the Python function: {self.func_config['func_schema']}, take the following user task:

        {self.func_config['prompt']}, and extract the relevant values into the input schema. Make sure to convert the user's input to match the input schema.

        Write Python code to set the input variables, import the Python function {self.func_config['func_name']} from {self.func_config['py_file']}.py inside the directory: {self.func_config['agent_dir']} (from {self.func_config['agent_dir']}.{self.func_config['py_file']} import {self.func_config['func_name']}), and input into the function.

        If the user task is NOT specific enough, please ask the user in natural language to provide more information.

        For example, if the user tasks you to do something but doesn't give specific values, you MUST ask for them explicitly - give your Final Answer asking for more info.
        Do NOT set or come up with the input variables yourself. You MUST ask the user for more information and extract the relevant inputs from the user.
        """
        return agent_task
    
    def get_json_task(self, prompt: str, use_time_prompt: bool = False) -> str:
        # JSON prompt for JSON agent. Extracts relevant fields from Objective.
        time_prompt = ""
        if use_time_prompt:
            time_prompt = ''''Note: If you have an objective_start_time or objective_end_time, 
            make sure to get the current time and explicitly fill in the field with the actual datetime string.'
        '''
        json_prompt = f'''Given the following user task: `{prompt}`, Examine the following JSON schema:

        {self.objective.schema}

        Create a JSON object that represents an instance of the `{self.objective.name}` class using information from the user task. Here is an example for a JSON that adheres to the schema:

        {self.objective.examples}

        Create a new JSON object and include every parameter from the class if it is required. Required: {self.objective.required} from the JSON schema.
        
        If data_mode not specified, use the default: {DEFAULT_DATA_MODE} and if classification_marking not specified use the default: {DEFAULT_MARKINGS}.
        '''

        return json_prompt
    
    def convo_code(self, prompt: str) -> Union[tuple, jsonify]:
        if self.agent_executor is None:
            self.init_convo_agent()

        self.validate_convo_agent()
        agent_task = self.get_agent_task(prompt)
        response = self.agent_executor.run(agent_task)

        responses = self.agent_history.agent_thoughts
        agent_responses = get_agent_thoughts(responses)

        code_block = extract_code_from_response_convo(agent_responses,
                                                      #try_output=response,
                                                      )
        
        responses += [response]
        responses += [f'Extracted code:\n{code_block}']

        valid_code = False
        if code_block is not None:
            valid_code = True
            filename = 'convo_code_' + str(self.id)
            save_code_to_file(code=code_block, filename=filename, dir_name='agent_dir')

        return jsonify({'response': responses}), valid_code
    
    def text_code(self, prompt: str) -> Union[tuple, jsonify]:
        self.init_agent()
        final_answer = self.agent.run(prompt)

        # Agent Logs
        logs = handler.log # gets verbose logs from agent LLM
        cleaned_logs = get_agent_thoughts(logs) # cleans logs
        self.save_response(cleaned_logs) # save logs for later debugging

        response = cleaned_logs + [final_answer]
        ## TODO: not sure if logic for stopping agent makes sense -- maybe we skip this
        if final_answer == "Agent stopped due to iteration limit or time limit.":
            return jsonify({'response': response})
        
        code_block = extract_code_from_response(cleaned_logs)
        response += [f'Extracted code:\n{code_block}']
        valid_code = False
        if code_block is not None:
            valid_code = True
            filename = 'text_code_' + str(self.id)
            save_code_to_file(code=code_block, filename=filename, dir_name='agent_dir')

        return jsonify({'response': response}), valid_code
    
    def validate_json(self, json_text: str) -> bool:
        pydantic_class = globals()[self.objective.name]
        parser = PydanticOutputParser(pydantic_object=pydantic_class)

        try:
            parsed_response = parser.parse(json_text)
        except OutputParserException as e:
            print(f"Failed to parse {self.objective.name}: {str(e)}")
            parsed_response = None

        return parsed_response is not None
    
    def validate_json_agent(self):
        assert self.json_agent_executor is not None
        assert self.memory is not None
        assert self.custom_json_history is not None
        assert self.agent is not None
        assert self.tools is not None

    def get_json_responses(self, prompt, responses):
        # This means previous issue with JSON in back-and-forth
        if self.prev_response is not None:
            prompt = f"Fix the following JSON: {self.prev_response} using the following additional information: " + prompt

        json_task = self.get_json_task(prompt, use_time_prompt=False)

        response = ""
        try:
            response = self.json_agent_executor.run(json_task)
        except ValueError as e:
            print(f"json_agent_executor error: {e}")

        agent_thoughts = self.custom_json_history.agent_thoughts

        if response == "":
            responses += agent_thoughts
            responses += ["Please provide more information regarding the fields above."]
            self.reset_thoughts()
            return responses, False
                
        try:
            json_text = extract_json(response)
        except ValueError as e:
            print(f"JSON object error: {e}")
            json_text = None

        if json_text is None:
            responses += agent_thoughts
            responses += ["Please provide more information regarding the fields above."]
            self.reset_thoughts()
            return responses, False
        
        valid_json = False
        if self.validate_json(json_text):
            responses += [f'Extracted valid JSON:\n{json_text}']
            valid_json = True
        else:
            self.prev_response = json_text
            responses += [f'Extracted invalid JSON:\n{json_text}']
            responses += ['Given the faulty fields above, give me more information to fix the JSON.']
            self.reset_thoughts()
            return responses, valid_json
        
        if valid_json:
            json_file = json.loads(json_text)
            save_json(json_file)
        
        self.prev_response = json_text # save in case of faulty "valid" JSONs
        return responses, valid_json
    
    def json_agent(self, prompt: str, custom_agent: bool = True):
        responses = []

        if self.objective is None:
            self.objective, confidence = get_objective(prompt)
            obj_response = ""
            # Shows confidence if less than 99.9% confident as a metric
            if 100 - confidence > 0.1:
                obj_response = (f"Objective extracted: {self.objective.obj_name}. "
                                f"Confidence: {confidence:.2f}%")
            else:
                obj_response = f"Objective extracted: {self.objective.obj_name}"
            responses += [obj_response]

        assert self.objective is not None

        # TODO: at some point custom will be default and init_json_agent will be deprecated...
        if custom_agent:
            self.init_custom_json_agent(use_custom_tools=False)
        else:
            self.init_json_agent()

        self.validate_json_agent()

        responses, valid_json = self.get_json_responses(prompt, responses)

        self.reset_thoughts()

        if self.app:  # If in Flask context
            return jsonify({'response': responses}), valid_json
        else:  # If in CLI context
            return {'response': responses}, valid_json
    
    def test_json(self, prompt: str, custom: bool = True):
        use_time = False
        responses = []

        self.objective, confidence = get_objective(prompt)
        obj_response = ""
        # Shows confidence if less than 99.9% confident as a metric
        if 100 - confidence > 0.1:
            obj_response = (f"Objective extracted: {self.objective.obj_name}. "
                            f"Confidence: {confidence:.2f}%")
        else:
            obj_response = f"Objective extracted: {self.objective.obj_name}"
        responses += [obj_response]

        assert self.objective is not None

        # TODO: at some point custom will be default and init_json_agent will be deprecated...
        if custom:
            self.init_custom_json_agent(use_custom_tools=False)
        else:
            self.init_json_agent()

        self.validate_json_agent()

        responses, valid_json = self.get_json_responses(prompt, responses)
        
        return responses, valid_json
    
    def init_api_agent(self):
        custom_tools = []

        ## Add list of API tools
        custom_tools += [get_skyfield_planets_tool]
        custom_tools += [get_planet_distance_tool]
        custom_tools += [get_latitude_longitude_tool]
        custom_tools += [get_skyfield_satellites_tool]
        custom_tools += [get_next_visible_time_for_satellite_tool]


        # TODO: template_with_history might need to be tuned/optimized
        if self.mistral:
            api_template = api_template_with_history_mistral_1 # or 2, 3, 4
        else:
            api_template = template_with_history_api
        
        self.custom_json_history = CustomPromptTemplate(
            template=api_template,
            tools=custom_tools,
            # prefix=JSON_PREFIX,
            # suffix=JSON_SUFFIX,
            agent_thoughts=self.agent_thoughts,
            input_variables=["input", "intermediate_steps", "history"]
        )

        llm_chain = LLMChain(llm=self.llm, prompt=self.custom_json_history)
        tool_names = [t.name for t in custom_tools]

        self.agent = LLMSingleActionAgent(
            llm_chain=llm_chain,
            output_parser=self.output_parser,
            stop=["\nObservation:"],
            allowed_tools=tool_names
        )

        num_memories = 3 # can play around with this (maybe 3, 4, 5, etc)
        self.memory = ConversationBufferWindowMemory(k=num_memories)
        self.tools = custom_tools

        max_iterations = 8
        self.json_agent_executor = AgentExecutor.from_agent_and_tools(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            memory=self.memory,
            max_iterations=max_iterations
        )

    def get_api_task(self, prompt: str) -> str:
        # Prompt for API agent. Extracts relevant fields from Objective.
        api_prompt = None
        if self.mistral:
            api_prompt = f'''[INST] Given the following user task: `{prompt}`, Use your Skyfield tools for planets to answer the user question. [/INST]
'''
        else:
            api_prompt = f'''Given the following user task: `{prompt}`, Use your Skyfield tools for planets to answer the user question.
'''
        return api_prompt
    
    def get_api_responses(self, prompt, responses):
        api_task = self.get_api_task(prompt)

        response = ""
        try:
            response = self.json_agent_executor.run(api_task)
        except ValueError as e:
            print(f"json_agent_executor error: {e}")

        agent_thoughts = self.custom_json_history.agent_thoughts

        if response == "":
            responses += agent_thoughts
            responses += ["I don't know."]
            self.reset_thoughts()
            return responses, False
        
        responses += [response]
        
        return responses, True ## TODO: This 2nd param is messy...
    
    def api_agent(self, prompt: str):
        responses = []

        self.init_api_agent()

        self.validate_json_agent() ## TODO: consider changing this name

        responses, valid_json = self.get_api_responses(prompt, responses)

        self.reset_thoughts()
        return jsonify({'response': responses}), valid_json
    
    def task_agent(self, prompt: str) -> Union[tuple, jsonify]:
        """Tasks LLM agent to process a given prompt/task."""
        if not prompt:
            return jsonify({'error': 'No task provided'}), 400
        
        if self.agent_mode == 'text-code':
            return self.text_code(prompt)
        elif self.agent_mode == 'convo-code':
            return self.convo_code(prompt)
        elif self.agent_mode == 'json-agent':
            return self.json_agent(prompt, custom_agent=True)
        elif self.agent_mode == 'api-agent':
            return self.api_agent(prompt)
        
        return self.text_code(prompt) # default

    def error_handler(self, e: Exception) -> jsonify:
        """Handle errors during request processing."""
        logging.error(f"Error occurred: {e}")
        return jsonify({'error': 'An error occurred while processing the request.'}), 500