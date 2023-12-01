import os
import openai

openai.api_key = os.environ['OPENAI_API_KEY']
openai.api_base="https://api.openai-hk.com/v1"

import re

def wf(str,dir):
    base_url=""
    #file_name="tesss.txt"
    Dir=base_url + dir
    f=open(Dir,"w",encoding='utf-8')
    f.write(str)

def zz(response):
    pattern=re.compile(r"<template>[\s\S]*<\/template>|<script>[\s\S]*<\/script>|<style>[\s\S]*<\/style>")      #[\s\S]*可以匹配任意字符串包括换行符
    #print(pattern)
    s=pattern.findall(response)
    s1=""
    for str in s:
        s1+=str+"\n"
    #print("解析结果:"+s1)
    #args=json.load(s1)
    return s1  

import os

def create_vue_project(project_name, path):
    """
    This function creates a new Vue project.

    Args:
        project_name (str): The name of the Vue project.
        path (str): The path where the project needs to be created.

    Returns:
        None
    """
    # Execute the command to create Vue project in the given path.

    cmd1 = f"cd {path}&&vue create -p demo {project_name}" 
    os.system(cmd1) 
    dir = path+project_name
    cmd3 = f"cd {dir}&&npm install element-ui -S&&npm install vue-router -S"
    os.system(cmd3)
    return project_name

def run_vue(dir):
    #cmd = f"cd {dir}&&npm run serve"
    para = dir
    cmd = f"a.bat {para}"
    os.system(cmd)
    #import subprocess
    #os.system("start cmd.exe&&dir")
    #subprocess.Popen('cmd.exe \C dir', shell=True)

def prj_select(base_url):
    print("请选择你操作：")
    print("1. Create a new project")
    print("2. Open a project")
    dict = {"1": "Create a new project", "2": "Open a project"}
    input1 = input("请输入你的选择：")
    if input1 == "1":
        input1 = input("请输入你的项目名称：")
        create_vue_project(input1,base_url)
        run_vue(base_url + input1)
    elif input1 == "2":
        input1 = input("请输入你的项目路径：")
        run_vue(base_url + input1)
    return base_url + input1 + "/"

def op(base_url):
    print("请选择你操作：")
    print("1. Create a new page")
    print("2. modify a page")
    dict = {"1": "write", "2": "modify"}
    input1 = input("请输入你的选择：")
    if input1 == "1":
        input1 = input("请输入你的页面名称：")
        url = base_url + "src/components/"
        page = input1 + ".vue"
        return url, page, dict["1"] 

def log(mem):                                   #也可以换种方式保存，每次对话结束后保存一次
    import datetime
    cur_time=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    f=open("log.txt","a",encoding='utf-8')
    f.write(cur_time+":\n")   
    f.write(mem)

def save_better(mem):                                  
    f=open("cov.txt","a",encoding='utf-8')
    str='------------------------------------------------------------------------'
    f.write(str+'\n')
    f.write(mem)

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory,ConversationSummaryBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

template="""You are a helpful assistant. You can help me by answering my questions. You can also ask me questions.

{memory}
Human:{human_input}
AI:"""

prompt=PromptTemplate(
    input_variables=["memory","human_input"], template=template
)
memory = ConversationBufferMemory(memory_key="memory")

llm_chain = LLMChain(
    llm=ChatOpenAI(temperature=0),
    memory=memory,
    #verbose=True,
    prompt=prompt
)

'''
functions = [
    {
        "name": "zz",
        "description": "将字符串中的代码和文件名字提取出来",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Various forms of code",
                },
                "file_name": {
                    "type": "string",
                    "description": "a file name of the code",
                }
            },
            "required": ["code"],
        },
    }
]
'''
# llm_chain.add_function(functions)


def llm(url,page,mode):
    dir = url
    input1 = "你好,请输入你的需求："
    response=""
    while True:
        #print("user:"+input1)
        if input1=="save":
            save_better(str(memory.load_memory_variables))
            continue
        if input1=="finish":
            log(str(memory.load_memory_variables))
            break

        #flag = re.search("write",input1)
        response=llm_chain.run(input1)
        if mode == "write":             #如何确定是追加 还是创建文件
            #content,Dir=zz(response)
            #Dir=re.findall(r"([A-Za-z0-9_]+\.[A-Za-z0-9_]+)",input1)
            d = dir + page 
            '''
            for dir in Dir:
                if dir!="vue.js":
                    #wf(response,dir)
                    d=dir
                    break
            if d=="":
                print("请输入文件名")
                continue
            '''
        content=zz(response)
        if(content!=""):
            wf(content,d)
            print("assisant：页面已经生成。")
            print("----------------------------------------------------------------------------------------------------------------")
            input1=input("user：")
            continue
        # if (re.search("json",input1)!=None):
        #    print("我开始解析了")
        #    #zz(response)
        print("assisant:"+response)
        print("----------------------------------------------------------------------------------------------------------------")
        input1=input("user：")

def main():
    base_url = prj_select("C:/Users/17466/Desktop/")
    url, page, mode = op(base_url)
    llm(url,page,mode)

main()