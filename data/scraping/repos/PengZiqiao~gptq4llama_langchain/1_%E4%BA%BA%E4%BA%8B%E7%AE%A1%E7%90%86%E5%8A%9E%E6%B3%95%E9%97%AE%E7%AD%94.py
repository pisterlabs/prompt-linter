import streamlit as st
from pathlib import Path
from langchain.vectorstores import Milvus
from langchain.prompts import load_prompt

import sys
sys.path.append('..')
from config import MILVUS_HOST, MILVUS_PORT
from model import streaming_generate, GPTQEmbeddings

TOP_K = 5

def clear():
    st.session_state["question"] = ""
    st.session_state["response"] = ""
    st.session_state["source"] = []


# 搜索相似文档
def similarity_search(question):
    # 使用 vector_db 的 similarity_search 搜索出相似文档
    docs = st.session_state["vector_db"].similarity_search(question, k=TOP_K)

    # 分别将k段结果的 page_content 和 metadata['source'] 拼接成 string
    source = []
    context = []
    for each in docs:
        source_path = Path(each.metadata['source'])
        source_name = source_path.stem
        content = each.page_content
        source.append(f'`FROM: 《{source_name}》`\n\n{content}')
        context.append(content)

    return '\n'.join(context), source

# QA查询
def query():
    # 获取问题
    question = st.session_state["question"]
    clear()

    # 获取关联文档
    context, source = similarity_search(question)
    st.session_state["source"] = source

    # 构建 prompt
    template = load_prompt("../prompts/QA.json")
    prompt = template.format(question=question, context=context)

    # 调用模型获得回复
    response = streaming_generate(prompt)
    st.session_state["response"] = response


# 初始化
if "vector_db" not in st.session_state:
    st.session_state["vector_db"] = Milvus(
        GPTQEmbeddings(),
        collection_name="HRDocs",
        connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT}
    )
    st.session_state["log"] = Path('log.txt')
    clear()

# 输入框
user_input = st.text_input("请输入您关于人事管理办法的问题，按回车发送", on_change=query, key="question")

if st.session_state["source"]:
    # 显示来源
    st.write('**参考内容：**')
    tabs = st.tabs([str(i+1) for i in range(TOP_K)])
    for i, source in enumerate(st.session_state["source"]):
        tabs[i].write(source)

    # 显示回复
    st.write('**回答：**')
    final_reply = st.empty()
    for each in st.session_state["response"]:
        reply = each.data
        final_reply.write(f':orange[{reply}]')
    
    # 写入日志
    text = st.session_state["log"].read_text()
    text += reply
    st.session_state["log"].write_text(text)