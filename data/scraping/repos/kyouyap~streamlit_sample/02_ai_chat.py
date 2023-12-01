import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import StreamlitCallbackHandler
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
from langchain.callbacks import get_openai_callback
from forex_python.converter import CurrencyRates
import datetime

from token_cost_process import TokenCostProcess,CostCalcAsyncHandler

def init_page():
    st.set_page_config(
        page_title="My Great ChatGPT",
        page_icon="🤗"
    )
    st.header("My Great ChatGPT 🤗")
    st.sidebar.title("Options")


def init_messages():
    init_content=f"""
    You are ChatGPT, a large language model trained by OpenAI, based on the GPT-4 architecture. Knowledge cutoff: 2021-09. Current date: {datetime.datetime.now().strftime("%Y-%m-%d")}.
    """
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    if clear_button or "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content=init_content)
        ]
        st.session_state.costs = []


def select_model():
    model = st.sidebar.radio("Choose a model:", ("GPT-3.5", "GPT-4"))
    if model == "GPT-3.5":
        model_name = "gpt-3.5-turbo"
    else:
        model_name = "gpt-4"

    # サイドバーにスライダーを追加し、temperatureを0から2までの範囲で選択可能にする
    # 初期値は0.0、刻み幅は0.1とする
    temperature = st.sidebar.slider("Temperature:", min_value=0.0, max_value=2.0, value=0.0, step=0.01)
    st.session_state.model_name = model_name
    return ChatOpenAI(temperature=temperature, model_name=model_name, streaming=True)

def show_massages(messages_container):
    messages = st.session_state.get('messages', [])
    with messages_container:
        for message in messages:
            if isinstance(message, AIMessage):
                with st.chat_message('assistant'):
                    st.markdown(message.content)
            elif isinstance(message, HumanMessage):
                with st.chat_message('user'):
                    st.markdown(message.content)
            else:  # isinstance(message, SystemMessage):
                st.write(f"System message: {message.content}")

def place_input_form(input_container, messages_container, llm):
    messages = st.session_state.get('messages', [])
    with input_container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_area(label='Message: ', key='input')
            submit_button = st.form_submit_button(label='Send')
        if submit_button and user_input:
            # 何か入力されて Submit ボタンが押されたら実行される
            st.session_state.messages.append(HumanMessage(content=user_input))
            with st.spinner("ChatGPT is typing ..."):
                response = llm(messages)
            st.session_state.messages.append(AIMessage(content=response.content))

def main():
    init_page()

    llm = select_model()
    init_messages()

    messages_container = st.container()  # メッセージ用のコンテナ
    input_container = st.container()  # 入力フォーム用のコンテナ

    place_input_form(input_container, messages_container, llm)
    show_massages(messages_container)
    # ユーザーの入力を監視




if __name__ == '__main__':
    main()