import os

import streamlit as st
from langchain.chat_models import ChatOpenAI

# タイトルを表示
st.title("simple chat")

# 入力を受け付け
prompt = st.text_input("What is up?")

# 入力があった場合、LLMを呼び出す
if prompt:
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    response = llm.predict(prompt)
    st.write(response)

# サイドバー
with st.sidebar:
    # APIキーの入力欄を表示
    os.environ["OPENAI_API_KEY"] = st.text_input("OpenAI API キー", type="password")
