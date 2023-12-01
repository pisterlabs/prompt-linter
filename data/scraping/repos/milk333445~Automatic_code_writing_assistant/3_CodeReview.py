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
import subprocess
import shutil


##相關函數
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

def code_rewiew_single(filename, logic_analysis, shared_knowledge, execution_results, code):
    PROMPT_TEMPLATE = """
    NOTICE
    Role: You are a professional software engineer, and your main task is to review the code. You need to ensure that the code conforms to the PEP8 standards, is elegantly designed and modularized, easy to read and maintain, and is written in Python 3.10 (or in another programming language).
    ATTENTION: Use '##' to SPLIT SECTIONS, not '#'. Output format carefully referenced "Format example".

    ## Code Review: Based on the following context and code, and following the check list, Provide key, clear, concise, and specific code modification suggestions, up to 7.
    ```
    1. Check 0: Is the code implemented as per the requirements?
    2. Check 1: Are there any issues with the code logic?
    3. Check 2: Does the existing code follow the "Data structures and interface definitions"?
    4. Check 3: Is there a function in the code that is omitted or not fully implemented that needs to be implemented?
    5. Check 4: Does the code have unnecessary or lack dependencies?
    6. Check 5: Does the code match the rest of the code?
    7. Check 6: How to fix the code if there is any error?
    ```

    ## Rewrite Code: {filename} Base on "Code Review" and the source code and the code Execution Results , rewrite code with triple quotes. Do your utmost to optimize THIS SINGLE FILE. 
    -----
    
    ## here is the logic analysis of the code:
    {logic_analysis}
    
    ## here is the shared knowledge of the code:
    {shared_knowledge}
    
    
    ## here is the code execution results:
    filename:{filename}
    
    {execution_results}
    
    if there is any error, please check the code and modify it.
    If any function or class is not defined, define it.
    
    ## the code that you have to review and rewrite: 
    filename:{filename}

    {code}
    
    ----
    ## Format example
    -----
    {format_example}
    -----

    """


    FORMAT_EXAMPLE = """

    ## Code Review
    1. The code ...
    2. ...
    3. ...
    4. ...
    5. ...

    ## Rewrite Code: {filename}
    ```python
    ...
    ```
    """
    llm = ChatOpenAI(temperature  =0.5, model_name = "gpt-3.5-turbo-16k")
    prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["filename", "logic_analysis", "shared_knowledge", "execution_results", "code", "format_example"])
    chain = LLMChain(llm=llm, prompt=prompt)
    raw_code_review = chain.run(filename = filename, logic_analysis = logic_analysis, shared_knowledge = shared_knowledge, execution_results = execution_results, code = code, format_example = FORMAT_EXAMPLE)
    print(raw_code_review)  
    return raw_code_review

#引入變數
if "code_history" not in st.session_state:
    st.session_state["code_history"] = ""
if "main_task" not in st.session_state:
    st.session_state["main_task"] = ""
if "data_api_design" not in st.session_state:
    st.session_state["data_api_design"] = ""
if "file_list" not in st.session_state:
    st.session_state["file_list"] = ""
    
if "main_mission" not in st.session_state:
    st.session_state["main_mission"] = ""


with st.sidebar:
    
    openai_api_key = st.text_input('OpenAI API Key', '', type="password")
os.environ['OPENAI_API_KEY'] = openai_api_key


st.title("Code Review")


st.info(f"最後相關程式碼會幫您整理到資料夾 workplace 中，請等我撰寫完成後至資料夾中查看相關程式檔。")


if st.button("下載相關套件"):
    with st.spinner('Download...'):     
        install_success = {}  # 用來記錄每個套件的安裝結果
        for package in st.session_state["python_package_name"]:
            result = subprocess.run(["pip", "install", package], capture_output=True, text=True)
            if result.returncode == 0:
                install_success[package] = "成功"
            else:
                install_success[package] = "失敗：" + result.stderr.strip()
        with st.expander("安裝結果"):
            for package, status in install_success.items():
                st.write(f"安裝{package}: {status}")
        
            


if st.button("進行Code Review"):
    with st.spinner('Code Reviewing...'):      
        folder_name = "workplace"
    # 檢查資料夾是否存在
        if os.path.exists(folder_name):
            # 如果資料夾存在，刪除資料夾及其內容
            shutil.rmtree(folder_name)
        os.mkdir(folder_name) 
        #執行程式檔
        folder_path = os.path.join(os.getcwd(), "first_version")
        execution_results = {}
        for file_name in reversed(st.session_state["file_list"]):
            file_path = os.path.join(folder_path, file_name)
            result = subprocess.run(['python', file_path], capture_output=True, text=True)
            execution_results[file_name] = {
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
        #輸出結果
        
        for snippet in st.session_state["code_history"]:
            code_review_object = ""
            code_review_object += f"```python\n{snippet.code}\n```\n"
            execution_result_str = ""
            execution_result_str += f'returncode: {execution_results[snippet.title]["returncode"]}\n'
            execution_result_str += f'stdout: {execution_results[snippet.title]["stdout"]}\n'
            execution_result_str += f'stderr: {execution_results[snippet.title]["stderr"]}\n'   
            #進行code review
            print('開始')
            raw_code_review = code_rewiew_single(snippet.title, st.session_state["logic_analysis"], st.session_state["share_knowledge"], execution_result_str, code_review_object)
            print('結束')
            code_review_content = raw_code_review.split("##")[1]
            #存起來
            st.session_state[f'{snippet.title}_Code_Review'] = code_review_content
            pure_code_after_review = CodeParser.parse_code(block="Rewrite Code", text=raw_code_review)
            with st.expander(f"## Code Review: {snippet.title}"):                  
                st.write(code_review_content)
                st.write("## Rewrite Code")
                st.code(pure_code_after_review, language="python")
            #更新code
            snippet.code = pure_code_after_review 
            # 將程式碼寫入 Python 檔案
            file_path_final = os.path.join(os.path.join(os.getcwd(), "workplace"), snippet.title)
            with open(file_path_final, "w") as f:
                f.write(snippet.code)
                
if st.button("查看歷史紀錄"):
    st.write("相關程式碼最後會幫您整理到資料夾中，請至資料夾中查看。")
    for snippet in st.session_state["code_history"]:
        with st.expander(f"## {snippet.title}"):
            #codereview
            if f'{snippet.title}_Code_Review' not in st.session_state:
                st.session_state[f'{snippet.title}_Code_Review'] = ""
                
            st.write(st.session_state[f'{snippet.title}_Code_Review'])
            st.code(snippet.code, language="python")