import os
import random

import openai
import streamlit as st
import streamlit_chat as stchat
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.streamlit import StreamlitCallbackHandler  # Does it work?
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)

openai.api_key = os.environ.get("OPENAI_API_KEY")

PAGE_TITLE = "AIアシスタント"

DEFAULT_SYSTEM_MESSAGE = """
あなたはウェブサービスの開発と運用を行う企業の社員です。ユーザは同僚で、あなたに仕事に関する質問を投げかけます。
アシスタントとして、ウェブサービスの開発と運用に役立つ回答を、できる限り根拠を示した上で返してください。
""".strip().replace(
    "\n", ""
)


def initialize_session_state():
    if "system_message" not in st.session_state:
        st.session_state.system_message = DEFAULT_SYSTEM_MESSAGE
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "human_messages" not in st.session_state:
        st.session_state.human_messages = []
    if "predicted_messages" not in st.session_state:
        st.session_state.predicted_messages = []
    if "avatar_seeds" not in st.session_state:
        st.session_state.avatar_seeds = random.sample(range(0, 100), k=2)


def load_new_conversation(system_message):
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(system_message),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template("{input}"),
        ]
    )
    llm = ChatOpenAI(
        streaming=True,
        callback_manager=CallbackManager(
            [StreamlitCallbackHandler(), StreamingStdOutCallbackHandler()]
        ),
        model_name="gpt-4",
        temperature=0,
        max_tokens=1024,
    )
    memory = ConversationBufferMemory(return_messages=True)
    conversation = ConversationChain(memory=memory, prompt=prompt, llm=llm)
    return conversation


def run():
    st.set_page_config(page_title=PAGE_TITLE)
    st.title(PAGE_TITLE)

    initialize_session_state()

    with st.form("input_form"):
        system_message = st.text_area(
            "AIへの指示を入力してください", placeholder=st.session_state.system_message
        )
        if not system_message:
            system_message = st.session_state.system_message

        human_message = st.text_area("質問を入力してください")
        submitted = st.form_submit_button("質問する")

        if submitted:
            if not human_message:
                st.error("質問を入力してください。")
                st.stop()

            with st.spinner("返事を待っています..."):
                if st.session_state.conversation:
                    conversation = st.session_state.conversation
                else:
                    conversation = load_new_conversation(system_message)

                predicted_message = conversation.predict(input=human_message)

                if (
                    not st.session_state.human_messages
                    or st.session_state.system_message != system_message
                ):
                    st.session_state.human_messages.append(
                        [human_message, system_message]
                    )
                    st.session_state.system_message = system_message
                else:
                    st.session_state.human_messages.append([human_message])

                st.session_state.predicted_messages.append(predicted_message)

                for i in range(len(st.session_state.predicted_messages) - 1, -1, -1):
                    stchat.message(
                        st.session_state.predicted_messages[i],
                        key=str(i),
                        avatar_style="thumbs",
                        seed=st.session_state.avatar_seeds[0],
                    )
                    for j, message in enumerate(st.session_state.human_messages[i]):
                        stchat.message(
                            message,
                            is_user=True,
                            key=str(i) + "_user" + str(j),
                            avatar_style="thumbs",
                            seed=st.session_state.avatar_seeds[1],
                        )
