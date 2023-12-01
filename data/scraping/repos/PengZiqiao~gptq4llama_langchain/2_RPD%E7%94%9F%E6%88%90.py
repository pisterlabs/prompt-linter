import streamlit as st
from pathlib import Path
from langchain.vectorstores import Milvus
from langchain.prompts import load_prompt

import sys
sys.path.append('..')
from model import streaming_generate

def clear():
    st.session_state["func_text"] = ""
    st.session_state["desc_text"] = ""
    st.session_state["response"] = ""

def sumbit(func, desc, contents):
    clear()
    contents = '，'.join([k for k, v in contents.items() if v])

    prompt = f"""Write a RPD(Product Requirments Document) about '{func}', the requirment is：{desc}。
    The PRD should contain {contents}。output in markdown(Not code) format, in Chinese language."""

    st.session_state["response"] = streaming_generate(prompt)

if "response" not in st.session_state:
    st.session_state["log"] = Path('log.txt')
    clear()



func_text = st.text_input("功能", placeholder="输入功能名，如「网站登录注册模块」", key='func_text')
desc_text = st.text_area("需求描述", placeholder="通过登录注册模块，建立账户体系，沉淀用户", key='desc_text')

st.write('包含内容')
contents={}
for label in ('需求背景', '需求目标', '用户使用流程', '功能概述', '功能详细说明', '实现逻辑'):
    contents[label] = st.checkbox(label, value=True)

st.button('提交', on_click=sumbit, args=[func_text, desc_text, contents])

if st.session_state["response"]:
    final_reply = st.empty()
    for each in st.session_state["response"]:
        reply = each.data
        final_reply.markdown(reply)

    # 写入日志
    text = st.session_state["log"].read_text()
    text += reply
    st.session_state["log"].write_text(text)
