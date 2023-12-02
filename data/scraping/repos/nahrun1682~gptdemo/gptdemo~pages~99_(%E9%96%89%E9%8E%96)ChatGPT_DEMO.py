import streamlit as st
from langchain.llms import OpenAI
import os
from dotenv import load_dotenv
import openai
#libsフォルダの中にあるsimple_chat_responseをimport
from libs.simple_chat_response import *

# .envファイルの読み込み
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

#ワイド表示
st.set_page_config(layout="wide")

#タイトルを表示
st.title('🦜ChatGPT DEMO')
st.subheader('まだmemory機能は未実装' )

model_name = st.radio(label='モデルを選択してね',
                 options=('gpt-3.5-turbo', 'gpt-4'),
                 index=0,
                 horizontal=True,
)
 
# 定数定義
USER_NAME = "user"
ASSISTANT_NAME = "assistant"

# チャットログを保存したセッション情報を初期化
if "chat_log" not in st.session_state:
    st.session_state.chat_log = []



user_msg = st.chat_input("ここにメッセージを入力")
if user_msg:
    # 以前のチャットログを表示
    for chat in st.session_state.chat_log:
        with st.chat_message(chat["name"]):
            st.write(chat["msg"])

    # 最新のメッセージを表示
    with st.chat_message(USER_NAME):
        st.write(user_msg)

    # アシスタントのメッセージを表示
    response = simple_response_chatgpt(model_name,user_msg)
    with st.chat_message(ASSISTANT_NAME):
        assistant_msg = ""
        assistant_response_area = st.empty()
        for chunk in response:
            # 回答を逐次表示
            tmp_assistant_msg = chunk["choices"][0]["delta"].get("content", "")
            assistant_msg += tmp_assistant_msg
            assistant_response_area.write(assistant_msg)

    # セッションにチャットログを追加
    st.session_state.chat_log.append({"name": USER_NAME, "msg": user_msg})
    st.session_state.chat_log.append({"name": ASSISTANT_NAME, "msg": assistant_msg})