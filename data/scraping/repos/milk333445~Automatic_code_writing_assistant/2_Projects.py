import streamlit as st
from langchain import OpenAI
from langchain import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from serpapi import GoogleSearch
from langchain.document_loaders import WebBaseLoader
import re
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
from langchain.schema import Document
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
import numpy as np
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, OpenAI, LLMChain
import os
import shutil

with st.sidebar:
    
    openai_api_key = st.text_input('OpenAI API Key', '', type="password")
os.environ['OPENAI_API_KEY'] = openai_api_key
#函數整理
import ast
#parser
class CodeParser:
    @classmethod
    def parse_block(cls, block: str, text: str) -> str:
        blocks = cls.parse_blocks(text)
        for k, v in blocks.items():
            if block in k:
                return v
        return ""
    @classmethod
    def parse_blocks(cls, text: str):
        #根據文本切割成多個塊
        blocks = text.split('##')
        
        #創建字典儲存每個block的標題跟內容
        block_dict = {}
        
        for block in blocks:
            #如果block不為空則繼續處理
            if block.strip() != "":
                block_title, block_content = block.split('\n', 1)
                block_dict[block_title.strip()] = block_content.strip()
        return block_dict
    @classmethod
    def parse_code(cls, block: str, text: str, lang: str = "") -> str:
        if block:
            text = cls.parse_block(block, text)
        pattern = rf'```{lang}.*?\s+(.*?)```'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            code = match.group(1)
        else:
            raise Exception(f"Error parsing code block: {block}")
        return code
    @classmethod
    def parse_str(cls, block: str, text: str, lang: str = ""):
        code = cls.parse_code(block, text, lang)
        code = code.split("=")[-1]
        code = code.strip().strip("'").strip("\"")
        return code
    
    @classmethod
    def parse_file_list(cls, block: str, text: str, lang: str = "")->list[str]:
        code = cls.parse_code(block, text, lang)
        pattern = r'\s*(.*=.*)?(\[.*\])'
        match = re.search(pattern, code, re.DOTALL)
        if match:
            tasks_list_str = match.group(2)
            tasks = ast.literal_eval(tasks_list_str)
        else:
            raise Exception
        return tasks      

class OutputParser:

    @classmethod
    def parse_blocks(cls, text: str):
        # 首先根据"##"将文本分割成不同的block
        blocks = text.split("##")

        # 创建一个字典，用于存储每个block的标题和内容
        block_dict = {}

        # 遍历所有的block
        for block in blocks:
            # 如果block不为空，则继续处理
            if block.strip() != "":
                # 将block的标题和内容分开，并分别去掉前后的空白字符
                block_title, block_content = block.split("\n", 1)
                # LLM可能出错，在这里做一下修正
                if block_title[-1] == ":":
                    block_title = block_title[:-1]
                block_dict[block_title.strip()] = block_content.strip()

        return block_dict

    @classmethod
    def parse_code(cls, text: str, lang: str = "") -> str:
        pattern = rf'```{lang}.*?\s+(.*?)```'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            code = match.group(1)
        else:
            raise Exception
        return code

    @classmethod
    def parse_str(cls, text: str):
        text = text.split("=")[-1]
        text = text.strip().strip("'").strip("\"")
        return text

    @classmethod
    def parse_file_list(cls, text: str) -> list[str]:
        # Regular expression pattern to find the tasks list.
        pattern = r'\s*(.*=.*)?(\[.*\])'

        # Extract tasks list string using regex.
        match = re.search(pattern, text, re.DOTALL)
        if match:
            tasks_list_str = match.group(2)

            # Convert string representation of list to a Python list using ast.literal_eval.
            tasks = ast.literal_eval(tasks_list_str)
        else:
            tasks = text.split("\n")
        return tasks



#寫程式
def code_writing(filename, context, past_code, data_api_design):
    llm = ChatOpenAI(temperature  =0.5, model_name = "gpt-3.5-turbo-16k")
    PROMPT_TEMPLATE = """
    NOTICE
    Role: You are a professional engineer; the main goal is to write PEP8 compliant, elegant, modular, easy to read and maintain Python 3.9 code (but you can also use other programming language)
    ATTENTION: Use '##' to SPLIT SECTIONS, not '#'. Output format carefully referenced "Format example".

    ## Code: {filename} Write code with triple quoto, based on the following list and context.
    1. Do your best to implement THIS ONLY ONE FILE. ONLY USE EXISTING API. IF NO API, IMPLEMENT IT.
    2. Requirement: Based on the context, implement one following code file, note to return only in code form, your code will be part of the entire project, so please implement complete, reliable, reusable code snippets
    3. Attention1: If there is any setting, ALWAYS SET A DEFAULT VALUE, ALWAYS USE STRONG TYPE AND EXPLICIT VARIABLE.
    4. Attention2: YOU MUST FOLLOW "Data structures and interface definitions". DONT CHANGE ANY DESIGN.
    5. Think before writing: What should be implemented and provided in this document?
    6. CAREFULLY CHECK THAT YOU DONT MISS ANY NECESSARY CLASS/FUNCTION IN THIS FILE.
    7. Do not use public member functions that do not exist in your design.

    -----
    # Context
    {context}
    -----
    ## Data structures and interface definitions
    {data_api_design}
    
    ## the past code that we have created
    ```python
    {past_code}
    ```
    ## Format example
    -----
    ## Code: {filename}
    ```python
    ## {filename}
    ...
    ```
    -----
    """
        
    prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["filename", "context", "data_api_design", "past_code"])
    chain = LLMChain(llm=llm, prompt=prompt)
    raw_code = chain.run(filename = filename, context = context, data_api_design = data_api_design, past_code = past_code)
    return raw_code

#寫函數迴圈
#寫函數
class CodeSnippet:
    def __init__(self, title, code):
        self.title = title
        self.code = code
def show_code(code_history):
    if not code_history:
        return ""
    code_history_str = ""
    for snippet in code_history:
        code_history_str += f"## {snippet.title}\n"
        code_history_str += f"```python\n{snippet.code}\n```\n"
    return code_history_str

def file_list_code_writing(file_list, content, data_api_design):
    file_list_str = "".join(f"{index + 1}.{filename}\n" for index, filename in enumerate(file_list))
    code_history = []
    folder_name = "first_version"
    # 檢查資料夾是否存在
    if os.path.exists(folder_name):
        # 如果資料夾存在，刪除資料夾及其內容
        shutil.rmtree(folder_name)
    #創建新資料夾
    os.mkdir(folder_name)
    for i in range(len(file_list)):
        code_history_str = show_code(code_history)
        raw_code = code_writing(file_list[i], content, code_history_str, data_api_design)
        code = OutputParser.parse_code(raw_code, "python")
        
        with st.expander(f"{file_list[i]} 已完成"):
            st.code(code)
        #寫入檔案
        with open(f"first_version/{file_list[i]}", "w") as f:
            f.write(code)
        st.session_state[file_list[i]] = code
        code_main = CodeSnippet(file_list[i], code)
        code_history.append(code_main)
        
    return code_history

#寫函數
def write_task(content):
    llm = ChatOpenAI(temperature  =0.5, model_name = "gpt-3.5-turbo-16k")
    #模板
    prompt_template = '''
    # Context
    {context}

    ## Format example
    {format_example}
    -----
    Role: You are a project manager; the goal is to break down tasks according to the content above, give a task list, and analyze task dependencies to start with the prerequisite modules
    Requirements: Based on the context, fill in the following missing information, note that all sections are returned in Python code triple quote form seperatedly. Here the granularity of the task is a file, if there are any missing files, you can supplement them
    Attention: Use '##' to split sections, not '#', and '## <SECTION_NAME>' SHOULD WRITE BEFORE the code and triple quote.

    ## Required Python third-party packages: Provided in requirements.txt format

    ## Required Other language third-party packages: Provided in requirements.txt format

    ## Full API spec: Use OpenAPI 3.0. Describe all APIs that may be used by both frontend and backend.

    ## Logic Analysis: Provided as a Python list[str, str]. the first is filename, the second is class/method/function should be implemented in this file. Analyze the dependencies between the files, which work should be done first

    ## Task list: Provided as Python list[str]. Each str is a filename, the more at the beginning, the more it is a prerequisite dependency, should be done first

    ## Shared Knowledge: Anything that should be public like utils' functions, config's variables details that should make clear first. 

    ## Anything UNCLEAR: Provide as Plain text. Make clear here. For example, don't forget a main entry. don't forget to init 3rd party libs.

    '''

    FORMAT_EXAMPLE = '''
    ---
    ## Required Python third-party packages
    ```python
    """
    flask==1.1.2
    bcrypt==3.2.0
    """
    ```

    ## Required Other language third-party packages
    ```python
    """
    No third-party ...
    """
    ```

    ## Full API spec
    ```python
    """
    openapi: 3.0.0
    ...
    description: A JSON object ...
    """
    ```

    ## Logic Analysis
    ```python
    [
        ("game.py", "Contains ..."),
    ]
    ```

    ## Task list
    ```python
    [
        "game.py",
    ]
    ```

    ## Shared Knowledge
    ```python
    """
    'game.py' contains ...
    """
    ```

    ## Anything UNCLEAR
    We need ... how to start.
    ---
    '''
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "format_example"])
    chain = LLMChain(llm=llm, prompt=prompt)
    main_task = chain.run(context=content, format_example=FORMAT_EXAMPLE)
    
    return main_task


        





st.title("Code Generator")
st.info(f"最後相關程式碼會幫您整理到資料夾 workplace 中，請等我撰寫完成後至資料夾中查看相關程式檔。")

if st.button("產生具體任務"):
    with st.spinner('Generating task...'):
        main_mission = write_task(st.session_state["raw_content"])
        logic_analysis = CodeParser.parse_file_list(block = "Logic Analysis", text = main_mission)
        st.session_state["logic_analysis"] = logic_analysis
        share_knowledge = CodeParser.parse_code(block = "Shared Knowledge", text = main_mission)
        st.session_state["share_knowledge"] = share_knowledge
        st.session_state["main_mission"] = main_mission
        with st.expander("任務"):
            st.write(st.session_state["main_mission"])




if st.button("🤖產生程式碼"):
    with st.spinner('Generating code...'):
        main_task = st.session_state["main_task"]
        data_api_design = st.session_state["data_api_design"]
        python_package_name = st.session_state["python_package_name"]
        seq_flow = st.session_state["seq_flow"]
        file_list = CodeParser.parse_file_list("Task list", st.session_state["main_mission"])
        st.session_state["file_list"] = file_list
        raw_content = st.session_state["raw_content"]
        #撰寫程式
        content = f"logic_analysis = {st.session_state['logic_analysis']}\nshare_knowledge = {st.session_state['share_knowledge']}"
        code_history = file_list_code_writing(file_list, content, data_api_design)
        st.session_state["code_history"] = code_history
        
        

if st.button("查看歷史紀錄"):
    st.write("相關程式碼最後會幫您整理到資料夾中，請至資料夾中查看。")
    
    with st.expander("具體任務"):
        st.write(st.session_state["main_mission"])
    for i in range(len(st.session_state["file_list"])):
        with st.expander(f"{st.session_state['file_list'][i]}"):
            st.code(st.session_state[st.session_state["file_list"][i]])