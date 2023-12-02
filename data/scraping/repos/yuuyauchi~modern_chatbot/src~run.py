import argparse

import openai
import streamlit as st
from models.langchain_model import LangChainChatBot
from models.llamaindex_model import LlamaindexChatBot

parser = argparse.ArgumentParser(description="使用するモデルを記述")
parser.add_argument("model_name")
parser.add_argument("model_id")
args = parser.parse_args()
model_id = None

if args.model_name == "llamaindex":
    query_engine = LlamaindexChatBot.deploy()
elif args.model_name == "langchain":
    chatbot = LangChainChatBot()
    chatbot.read_data()
    chatbot.preprocess()
    query_engine = chatbot.generate_engine()
elif args.model_name == "chatgpt":
    model_id = args.model_id
else:
    raise NameError("モデル名の引数が間違っている或いは入力されていません。")

if "messages" not in st.session_state:
    st.session_state.messages = []
st.title("Let's try!")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("質問を入力してください。"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("回答を生成中..."):
        if args.model_name in ["llamaindex", "langchain"]:
            response = query_engine.query(prompt)
        elif args.model_name == "chatgpt":
            response = openai.ChatCompletion.create(
                model=model_id, messages=[{"role": "user", "content": prompt}]
            )
            response = response.to_dict_recursive()["choices"][0]["message"]["content"]

    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
