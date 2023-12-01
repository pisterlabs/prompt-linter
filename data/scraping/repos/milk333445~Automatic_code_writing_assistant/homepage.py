import streamlit as st
st.set_page_config(
    page_title="Automatic code writing robot",
    page_icon="🧊"
    
)

import streamlit as st
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, OpenAI, LLMChain
import time
import os
import ast
import re
from time import sleep
from PIL import Image
import subprocess
from pathlib import Path
import json, logging, pickle, sys, shutil, copy
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
#函數設定
def generate_task(task):
    llm = ChatOpenAI(temperature  =0.5, model_name = "gpt-3.5-turbo-16k")
    prompt_template = """
    # Context
    {context}

    ## Format example
    {format_example}
    -----
    Role: You are an architect; the goal is to design a SOTA PEP8-compliant python system; make the best use of good open source tools
    Requirement: Fill in the following missing information based on the context, note that all sections are response with code form separately
    Max Output: 8192 chars or 2048 tokens. Try to use them up.
    Attention: Use '##' to split sections, not '#', and '## <SECTION_NAME>' SHOULD WRITE BEFORE the code and triple quote.

    ## Implementation approach: Provide as Plain text. Analyze the difficult points of the requirements, select the appropriate open-source framework.

    ## Python package name: Please provide the necessary Python packages in the form of a Python list[str], using triple quotes for Python, and keep it concise and clear.

    ## File list: Provided as Python list[str], the list of ONLY REQUIRED files needed to write the program(LESS IS MORE!). Only need relative paths, comply with PEP8 standards. ALWAYS write a main.py or app.py here

    ## Data structures and interface definitions: Use mermaid classDiagram code syntax, including classes (INCLUDING __init__ method) and functions (with type annotations), CLEARLY MARK the RELATIONSHIPS between classes, and comply with PEP8 standards. The data structures SHOULD BE VERY DETAILED and the API should be comprehensive with a complete design. 

    ## Program call flow: Use sequenceDiagram code syntax, COMPLETE and VERY DETAILED, using CLASSES AND API DEFINED ABOVE accurately, covering the CRUD AND INIT of each object, SYNTAX MUST BE CORRECT.

    ## Anything UNCLEAR: Provide as Plain text. Make clear here.

    """

    FORMAT_EXAMPLE ="""
    ---
    ## Implementation approach
    We will ...

    ## Python package name
    ```python
    [
        "numpy",
    ]
    ```

    ## File list
    ```python
    [
        "main.py",
    ]
    ```

    ## Data structures and interface definitions
    ```mermaid
    classDiagram
        class Game{
            +int score
        }
        ...
        Game "1" -- "1" Food: has
    ```

    ## Program call flow
    ```mermaid
    sequenceDiagram
        participant M as Main
        ...
        G->>M: end game
    ```

    ## Anything UNCLEAR
    The requirement is clear to me.
    ---
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "format_example"])
    chain = LLMChain(llm=llm, prompt=prompt)
    raw_content = chain.run(context = task, format_example = FORMAT_EXAMPLE)
    data_api_design = CodeParser.parse_code(block="Data structures and interface definitions", text=raw_content)
    python_package_name = CodeParser.parse_file_list(block="Python package name", text=raw_content)
    seq_flow = CodeParser.parse_code(block="Program call flow", text=raw_content)
    file_list = CodeParser.parse_file_list(block="File list", text=raw_content)
    main_task = raw_content.split("##")[1]
    unclear = raw_content.split("##")[-1]

    return main_task, data_api_design, python_package_name, seq_flow, file_list, raw_content, unclear
def mermaid_to_png(mermaid_code, output_file, width=2048, height=2048):
    # Write the Mermaid code to a temporary file
    tmp = Path(f'{output_file}.mmd')
    tmp.write_text(mermaid_code, encoding='utf-8')

    output_file = f'{output_file}.png'
    # Call the mmdc command to convert the Mermaid code to a SVG
    mmdc_path = shutil.which('mmdc.cmd')
    subprocess.run([mmdc_path, '-i', str(tmp), '-o', output_file, '-w', str(width), '-H', str(height)]) 
#app架構



with st.sidebar:
    
    st.write("## 請輸入以下資料:")
    openai_api_key = st.text_input('OpenAI API Key', '', type="password")
    
    
    
if "onenai_api_key" not in st.session_state:
    st.session_state["openai_api_key"] = ""    
os.environ['OPENAI_API_KEY'] = openai_api_key
#儲存變數
if "main_task" not in st.session_state:
    st.session_state["main_task"] = ""
if "data_api_design" not in st.session_state:
    st.session_state["data_api_design"] = ""
if "python_package_name" not in st.session_state:
    st.session_state["python_package_name"] = ""
if "seq_flow" not in st.session_state:
    st.session_state["seq_flow"] = ""
if "file_list" not in st.session_state:
    st.session_state["file_list"] = ""
if "raw_content" not in st.session_state:
    st.session_state["raw_content"] = ""
if "unclear" not in st.session_state:
    st.session_state["unclear"] = ""
if "task" not in st.session_state:
    st.session_state["task"] = ""
if "detailed_goal" not in st.session_state:
    st.session_state["detailed_goal"] = ""

st.session_state['openai_api_key'] = os.environ['OPENAI_API_KEY']

#app架構
st.title("🤖Automated Task Completion")
task = st.text_input("您想要解決的任務是什麼？")

with st.expander("更詳細的目標"):
    detailed_goal = st.text_area("請在此輸入更詳細的內容：")


if st.button("🤖開始分析"):
    if task:
        st.session_state["task"] = task
        st.session_state["detailed_goal"] = detailed_goal
        if not openai_api_key.startswith('sk-'):
                st.warning('Please enter your OpenAI API key in the sidebar')
        else:
            
            with st.spinner('Generating...'):
                main_task, data_api_design, python_package_name, seq_flow, file_list, raw_content, unclear = generate_task(task + "\n" + "關於此任務的一些補充敘述: " + detailed_goal)
                st.session_state["main_task"] = main_task
                st.session_state["data_api_design"] = data_api_design
                st.session_state["python_package_name"] = python_package_name
                st.session_state["seq_flow"] = seq_flow
                st.session_state["file_list"] = file_list
                st.session_state["raw_content"] = raw_content
                st.session_state["unclear"] = unclear
                st.write("## 👇🏻我的任務內容如下:")
                st.info(main_task)
                #stmd.st_mermaid(data_api_design)
                st.write("## 👇🏻我的資料結構設計如下:")
                mermaid_to_png(data_api_design, "data_api_design")
                image = Image.open('data_api_design.png')
                st.image(image, caption='Data structures and interface definitions')
                st.write("## 👇🏻文件列表:")
                st.write(file_list)
                st.write("## 👇🏻不清楚的地方:")
                st.info(unclear)
    else:
        st.warning("請輸入任務要求")

with st.expander("查看歷史紀錄"):
    st.write("## 👇🏻原始問題為:")
    st.info(st.session_state["task"])
    st.write("## 👇🏻我的任務內容如下:")
    st.info(st.session_state["main_task"])
    st.write("## 👇🏻必要安裝套件如下:")
    st.write(st.session_state["python_package_name"])
    #stmd.st_mermaid(data_api_design)
    st.write("## 👇🏻我的資料結構設計如下:")
    try:
        image = Image.open('data_api_design.png')
        st.image(image, caption='Data structures and interface definitions')
    except:
        st.write("尚未生成圖片")
    st.write("## 👇🏻文件列表:")
    st.write(st.session_state["file_list"])
    st.write("## 👇🏻不清楚的地方:")
    st.info(st.session_state["unclear"])
    