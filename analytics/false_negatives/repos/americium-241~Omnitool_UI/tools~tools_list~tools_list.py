import streamlit as st
from streamlit_elements import elements, mui, html

import socket
import os 

from typing import Dict, Union
import sys
import io

import autogen
from langchain.tools import WikipediaQueryRun
from langchain.utilities import WikipediaAPIWrapper
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import HuggingFaceHub

from tools.base_tools import Ui_Tool
from storage.logger_config import logger
from tools.utils import evaluate_function_string
from tools.tool_manager import ToolManager

Local_dir=dir_path = os.path.dirname(os.path.realpath(__file__))
#!!! Carefull in this file, only used to define tools as functions 
# (with a docstring)and tools as base_tool child classes to link ui 
# or internal app logic


class Code_sender(Ui_Tool):
  
    
    name = 'Code_sender'
    icon= '💻'
    title= 'Code Sender'
    description="Allow you to send python script code to complete the task and get the result of the code"

    def _run(self,code): # callback not working, is still inputed by the run( ... callabacks)
       
        try:
            logger.info("CODE executed : %s",code)
            st.session_state.executed_code.append(code)
            return 'Success, returns : ' + str(exec(code, globals(), locals()))
        except Exception as e:
            return f"An error occurred while executing the code: {e}"
        
        
    def _ui(self):

        def checkstate(value):
            st.session_state.autorun_state=value['target']['checked']
           
        with mui.Accordion():
            with mui.AccordionSummary(expandIcon=mui.icon.ExpandMore):
                mui.Typography("Options")

            with mui.AccordionDetails():
              
                mui.FormControlLabel(
                    control=mui.Checkbox(onChange=checkstate,checked= st.session_state.autorun_state),
                    label="Auto run")
           

               

class stable_diffusion(Ui_Tool):

    name = 'stable_diffusion'
    title='Stable diffusion'
    icon= '🖼️'
    description=  'This tool allow for the creation of an image from a text input : question'


    def _run(self,question):
        import torch
        from diffusers import StableDiffusionPipeline
        torch_dtype=torch.float16
        pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5",force_download=True,resume_download= False)
        pipe = pipe.to(st.session_state.diffusion_option[0]) # pipe.to('cuda') #si nvidia gpu 

        prompt = question
        image = pipe(prompt).images[0]
        st.image(image)

        return 'Success '     
    def _ui(self):
        with mui.Accordion():
            with mui.AccordionSummary(expandIcon=mui.icon.ExpandMore):
                mui.Typography("Options")
            with mui.AccordionDetails():

                # Initialize a session state variable to store the selected items
                if "diffusion_option" not in st.session_state:
                    st.session_state.diffusion_option = ['cpu']  # Initialized as an array

                def handle_selection_change(state,value):
                    st.session_state.diffusion_option = [value['props']['value']]
                
                options = ["cpu", "gpu"]

                with elements("multi_select_element"):
                    # Creating a label for the select component
                    mui.Typography("Select running option")

                    # Creating the multi-choice select box
                    with mui.Select(multiple=True, value=st.session_state.diffusion_option, onChange=handle_selection_change):
                        for option in options:
                            mui.MenuItem(value=option, children=option)

class hugging_call(Ui_Tool):

    name = 'hugging_call'
    title='Hugging call'
    icon= '🤗'
    description=  'This tool allow the call for a hugging_face NLP that returns the answer, input: question'


    def _run(self,question):
        template = """Question: {question}

        Answer: Let's think step by step."""
       
        prompt = PromptTemplate(template=template, input_variables=["question"])
        # "openai-gpt"#"google/flan-t5-xxl"  # See https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads for some other options
        llm = HuggingFaceHub(
        repo_id=st.session_state.hugging_option[0], model_kwargs={"temperature": 0.1, "max_length":  st.session_state.hugging_length_option }
        )
        llm_chain = LLMChain(prompt=prompt, llm=llm)

        return (llm_chain.run(question))
    
    def _ui(self):
        with mui.Accordion():
            with mui.AccordionSummary(expandIcon=mui.icon.ExpandMore):
                mui.Typography("Options")
            with mui.AccordionDetails():

                # Initialize a session state variable to store the selected items
                if "hugging_option" not in st.session_state:
                    st.session_state.hugging_option = ['google/flan-t5-xxl']  # Initialized as an array

                if "hugging_length_option" not in st.session_state:
                    st.session_state.hugging_length_option = 500  # Initialized as an array

                def handle_selection_change(state,value):
                    st.session_state.hugging_option = [value['props']['value']]

                def handle_selection_length_change(value):
                    st.session_state.hugging_length_option = value['target']['value']

                options =["databricks/dolly-v2-3b","Writer/camel-5b-hf","Salesforce/xgen-7b-8k-base","tiiuae/falcon-40b","openai-gpt","google/flan-t5-xxl"]
                with elements("multi_select_element"):
                    # Creating a label for the select component
                    mui.Typography("Select your options:")

                    # Creating the multi-choice select box
                    with mui.Select(multiple=True, value=st.session_state.hugging_option, onChange=handle_selection_change):
                        for option in options:
                            mui.MenuItem(value=option, children=option)

                with elements("value_input_element"):
                # Creating a label for the numeric input
                    mui.Typography("Enter max_length")
                    # Numeric input
                    mui.TextField(
                        label="",
                        variant="outlined",
                        type="number",  # This makes it a numeric input
                        fullWidth=True,
                        defaultValue=st.session_state.hugging_length_option,
                        onChange=handle_selection_length_change
                    )




class db_query(Ui_Tool):

    name = 'db_query'
    title='Db Query'
    icon= '💽'
    description=  'Send the user question to an agent that will explore the database. query: question NOT a SQL query'


    def _run(self,query):


        from langchain.agents import create_sql_agent
        from langchain.agents.agent_toolkits import SQLDatabaseToolkit
        from langchain.sql_database import SQLDatabase
        from langchain.llms.openai import OpenAI
        from langchain.agents.agent_types import AgentType
        from langchain.chat_models import ChatOpenAI

        db = SQLDatabase.from_uri(st.session_state.db_option)
        toolkit = SQLDatabaseToolkit(db=db, llm=ChatOpenAI(temperature=0))

        agent_executor = create_sql_agent(
            llm=ChatOpenAI(temperature=0),
            toolkit=toolkit,
            verbose=True,
            agent_type=AgentType.OPENAI_FUNCTIONS,
        )

        answer =agent_executor.run(query)
        return 'Tool run, returns : '+answer



    def _ui(self):
        with mui.Accordion():
            with mui.AccordionSummary(expandIcon=mui.icon.ExpandMore):
                mui.Typography("Options")
            with mui.AccordionDetails():

                # Initialize a session state variable to store the selected items
                if "db_option" not in st.session_state:
                    st.session_state.db_option = 'sqlite:///'+Local_dir+'\\..\\..\\storage\\app_session_history.db'

                def handle_selection_change(value):
                    st.session_state.db_option = value['target']['value']
                 

                with elements("value_input_element"):
                    # Creating a label for the numeric input
                    mui.Typography("Enter database path")
                    # Numeric input
                    mui.TextField(
                        label="",
                        variant="outlined",
                        fullWidth=True,
                        defaultValue= st.session_state.db_option,
                        onChange=handle_selection_change
                    )


class socket_com(Ui_Tool):

    name = 'socket_com'
    title='Socket Com'
    icon= '📨'
    description=  'Send a message to the client. Input: message'


    def _run(self,question):

        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        host = socket.gethostname()  # get name of local machine
        port = st.session_state.socket_option

        def connect_to_host(host, port):
            '''Connect to a specified host and port.'''
            try:
                client_socket.connect((host, port))
                st.info(f"Connected to {host}:{port}")
            except socket.error as e:
                st.error(f"Failed to connect to {host}:{port}. Error: {e}")
                raise

        def close_connection():
            '''Close the client socket connection.'''
            try:
                client_socket.close()
                st.info("Connection closed successfully")
            except socket.error as e:
                st.error(f"Error while closing connection: {e}")
                raise

        def send_message(msg):
            '''Send a message to the connected host.'''
            try:
                client_socket.sendall(msg.encode())  # Use sendall to ensure complete data is sent
                st.info(f"Sent message: {msg}")
            except socket.error as e:
                st.error(f"Failed to send message: {msg}. Error: {e}")
                raise

        def receive_message():
            '''Receive a message from the connected host.'''
            try:
                data = client_socket.recv(1024).decode()
                st.info(f"Received message: {data}")
                return data
            except socket.error as e:
                st.error(f"Failed to receive message. Error: {e}")
                raise

        try:
            connect_to_host(host, port)
            send_message(question)
            response = receive_message()
            return f'Error is: {response}'
        finally:
            close_connection()

    def _ui(self):
        with mui.Accordion():
            with mui.AccordionSummary(expandIcon=mui.icon.ExpandMore):
                mui.Typography("Options")
            with mui.AccordionDetails():

                # Initialize a session state variable to store the selected items
                if "socket_option" not in st.session_state:
                    st.session_state.socket_option = 55555

                def handle_selection_change(value):
                    st.session_state.socket_option = value['target']['value']
                    print('Socket ;',st.session_state.socket_option)
                 
                  

                with elements("value_input_element"):
                    # Creating a label for the numeric input
                    mui.Typography("Enter port number")
                    # Numeric input
                    mui.TextField(
                        label="",
                        variant="outlined",
                        type="number",  # This makes it a numeric input
                        fullWidth=True,
                        defaultValue=st.session_state.socket_option,
                        onChange=handle_selection_change
                    )


def autogen_code_writer(question):
    '''This tool gets the input from autogen_plan and writes a python code that have to be sent to code_exec tool question is one simple string'''
    prompt_writer ="""You should create a python code that precisely solves the problem asked. Always make one single python snippet and assume that exemples should be made with randomly generated data rather than loaded ones.
    format : The python code should be formated as ```python \n ... \n ``` 
    ALWAYS finish your answer by \n TERMINATE"""
    # create an AssistantAgent named "assistant"
    code_writer = autogen.AssistantAgent(
        name="code_writer",
        human_input_mode="NEVER",
        llm_config=st.session_state.autogen_llm_config,
        system_message=prompt_writer,
        is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
        )
    user_proxy = autogen.UserProxyAgent(
        name="user_proxy",
        max_consecutive_auto_reply=0,  # terminate without auto-reply
        human_input_mode="NEVER",
        )

    # the assistant receives a message from the user_proxy, which contains the task description
    message = user_proxy.initiate_chat(
        code_writer,
        message=question,
    )
    return user_proxy._oai_messages[list(user_proxy._oai_messages.keys())[0]][1]['content']



def autogen_code_exec(question):
    '''This tool extract the code from question when formatted as  ``` \n python code \n ``` and will execute it'''
    class ExecUserProxyAgent(autogen.UserProxyAgent):
        def __init__(self, name: str, **kwargs):
            super().__init__(name, **kwargs)
            self._locals = {}

        def generate_init_message(self, *args, **kwargs) -> Union[str, Dict]:
            return super().generate_init_message(*args, **kwargs) 

        def run_code(self, code, **kwargs):
            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()

            # Redirecting stdout and stderr
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture

            exitcode = 0
            result = None
            try:
                exec(code, self._locals)
            except Exception as e:
                exitcode = 1
                stderr_capture.write(str(e))
                #bpy.ops.object.select_all(action='SELECT')  # Select all objects in the scene
                #bpy.ops.object.delete()

            # Reset stdout and stderr
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__

            log = stdout_capture.getvalue() + stderr_capture.getvalue()
            
            return exitcode, log, None
        
    # create a UserProxyAgent instance named "user_proxy"
    code_executor = ExecUserProxyAgent(
        name="code_executor",
        human_input_mode="NEVER",
        system_message="""You simply receive a message with code that will be executed, you can discuss ways to improve this code and return a better version if needed
        ALWAYS finish your answer by \n TERMINATE""",
    
        is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
        code_execution_config={
            "work_dir": "coding",
            "use_docker": False,  # set to True or image name like "python:3" to use docker
        },

    )
    user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    max_consecutive_auto_reply=0,  # terminate without auto-reply
    human_input_mode="NEVER",
    )
  
    message = user_proxy.initiate_chat(
        code_executor,
        message=question,
    )

    return user_proxy._oai_messages[list(user_proxy._oai_messages.keys())[0]][1]['content']



def autogen_plan(question): 
    '''This tool takes as input the fully detailed context of user question in order to construct a plan of action, always call at first or when confused'''

    autogen_planner = autogen.AssistantAgent(
        name="autogen_plan",
        system_message="""NEVER WRITE PYTHON CODE. Your job is to improve the question you receive by making it a clear step by step problem solving . Never write code, only explanations.
        Be precise and take into account that a LLM is reading your output to follow your instructions. You should remind in your answer that your message is intended for the code_writer 
        ALWAYS finish your answer by \n TERMINATE""",
        llm_config=st.session_state.autogen_llm_config,
        human_input_mode="NEVER",
        is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    )


    user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    max_consecutive_auto_reply=0,  # terminate without auto-reply
    human_input_mode="NEVER",
    )
  
    # the assistant receives a message from the user_proxy, which contains the task description
    message = user_proxy.initiate_chat(
        autogen_planner,
        message=question,
    )

    return user_proxy._oai_messages[list(user_proxy._oai_messages.keys())[0]][1]['content']

def powershell_terminal(command):
    '''send powershell commands to be executed separated with ; (each commands are an independant subprocess)'''
    import subprocess
    process = subprocess.Popen(["powershell", command], stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd='./workspace')
    result, error = process.communicate()

    # Decode result and error
    try:
        result_str = result.decode('utf-8')
    except UnicodeDecodeError:
        result_str = result.decode('cp1252', errors='replace')

    try:
        error_str = error.decode('utf-8')
    except UnicodeDecodeError:
        error_str = error.decode('cp1252', errors='replace')

    # Check return code
    if process.returncode != 0:
        return 'Error (code: {}): {}'.format(process.returncode, error_str)
    else:
        # You might still want to return any "info" or "progress" messages from stderr even if the operation succeeded.
        # Thus, you can check if error_str is not empty and append it to the success message.
        additional_info = '\nInfo from stderr: ' + error_str if error_str.strip() else ''
        return 'Success, code returns: ' + result_str + additional_info





def dataframe_query(query):
    """allow you to query the avaible dataframe in the workspace"""
    import pandas as pd
    from langchain.llms import OpenAI
    import os
    workspace= Local_dir+'\..\..\workspace'
    #create list of dataframes
    df_list = []
    for file in os.listdir(workspace):
        if file.endswith(".csv"):
            df_list.append(pd.read_csv(os.path.join(workspace, file)))
        elif file.endswith(".xlsx"):
            df_list.append(pd.read_excel(os.path.join(workspace, file)))
    

    agent = create_pandas_dataframe_agent(OpenAI(temperature=0), df_list, verbose=True)
    r=agent.run(query)
    return r
    
def browser_search(query):
    """allow you to navigate using the browser, provide url or keyword and instructions"""
    import subprocess
    response = str(subprocess.run(['python', Local_dir+'\..\browser_tool.py', query], text=True, capture_output=True))
    return response 


def wiki_search(query):
    """allow you to query the wikipedia api to get information about your query"""
    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    return wikipedia.run(query)



def make_tool(toolcode):
    """
    Send your toolcode, which is a string function with a docstring inside formatted as
    def toolname(input):
        \''' MANDATORY docstring to describe tool execution \''' 
        tool execution code
        return "Success"  
    """
    runs_without_error, has_doc, tool_name = evaluate_function_string(toolcode)

    if not runs_without_error:
        return f"Error: {runs_without_error}"

    if not has_doc:
        return "Error: Function must have a docstring."

    if not tool_name:
        return "Error: Could not extract tool name from the provided code."

    tools_path = os.path.join(Local_dir, f"{tool_name}.py")
    
    with open(tools_path, "w") as f:
        f.write('\n' + toolcode)

    
    st.session_state.tool_manager = ToolManager()
    st.session_state.tool_list = st.session_state.tool_manager.structured_tools
    return 'Success'