import streamlit as st
import pandas as pd
import numpy as np
from streamlit_app import check_password
import openai
import os
from langchain.chat_models import AzureChatOpenAI
from langchain.llms import AzureOpenAI
from langchain.chat_models import ChatVertexAI


from typing import List, Dict, Callable
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    BaseMessage,
)

from langchain.agents import Tool
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.agents import load_tools
from agentclass import DialogueAgent, DialogueAgentWithTools, DialogueSimulator
from translate import translate_text

# from callbackhandler import SimpleStreamlitCallbackHandler

# streamling用のもの
# handler = SimpleStreamlitCallbackHandler()

BASE_URL = st.secrets["OPENAI_API_BASE"]
API_KEY = st.secrets["API_KEY"]


with st.sidebar.form("パラメータ指定", clear_on_submit=False):
    topic = st.text_input(
        "議論内容：",
        "ヴィーガンレザーについて",
        key="placeholder",
    )
    roles = st.multiselect(
        "議論の役割選択",
        [
            "エンジニア",
            "デザイナー",
            "研究者",
            "弁護士",
            "ファイナンシャル・アドバイザー",
            "会計士",
            "企業アナリスト",
            "投資家",
            "皮肉屋",
            "悲観論者",
            "楽観論者",
        ],
        ["エンジニア", "投資家", "悲観論者"],
    )
    # model選択
    # option = st.selectbox("モデル選択", ("gpt-3.5", "gpt-4", "PaLM2"))
    # if option == "gpt3.5":
    DEPLOYMENT_NAME = st.secrets["OPENAI_API_MODEL_NAME_35"]
    # else:
    #    DEPLOYMENT_NAME = st.secrets["OPENAI_API_MODEL_NAME"]

    submitted = st.form_submit_button("開始！")


llm = AzureChatOpenAI(
    openai_api_base=BASE_URL,
    openai_api_version="2023-03-15-preview",
    deployment_name=DEPLOYMENT_NAME,
    openai_api_key=API_KEY,
    openai_api_type="azure",
    temperature=0.0,
    # streaming=True,
)


# st.set_page_config(page_title="LLM同士の対話", page_icon="🌍", layout="wide")


def generate_agent_description(name):
    agent_specifier_prompt = [
        agent_descriptor_system_message,
        HumanMessage(
            content=f"""{conversation_description}
            Please reply with a creative description of {name}, in {word_limit} words or less.
            Speak directly to {name}.
            Give them a point of view.
            Do not add anything else."""
        ),
    ]
    agent_description = llm(agent_specifier_prompt).content
    # st.write(agent_description)
    return agent_description


# @title generate_system_message(name, description, tools)
def generate_system_message(name, description, tools):
    # Your goal is to explain your conversation partner of your point of view.
    return f"""{conversation_description}

Your name is {name}.
Your description is as follows: {description}
Your goal is to persuade your conversation partner of your point of view.
DO look up information with your tool.
DO cite your sources.

DO NOT fabricate fake citations.
DO NOT cite any source that you did not look up.

Do not add anything else.

Stop speaking the moment you finish speaking from your perspective.

Speak in English. 
Always show your sources,urls etc.
"""


def select_next_speaker(step: int, agents: List[DialogueAgent]) -> int:
    idx = (step) % len(agents)
    return idx


# if check_password():
if True:
    pd.options.display.precision = 1

    names = {}

    for role in roles:
        names[role] = ["wikipedia", "arxiv", "ddg-search", "google-search"]

    # names = {
    #    "lawyer": ["human", "ddg-search", "wikipedia"],
    #    "financial advisor": ["human", "ddg-search", "wikipedia"],
    #    "accountant": ["human", "ddg-search", "wikipedia"],
    #    "researcher": ["human", "arxiv", "ddg-search", "wikipedia"],
    # }

    word_limit = 50  # word limit for task brainstorming

    if submitted:
        # @title generate_agent_description(name)
        conversation_description = f"""Here is the topic of conversation: {topic}
        The participants are: {', '.join(names.keys())}"""

        agent_descriptor_system_message = SystemMessage(
            content="You can add detail to the description of the conversation participant."
        )

        agent_descriptions = {name: generate_agent_description(name) for name in names}

        with st.sidebar.expander("役割の説明"):
            for name, description in agent_descriptions.items():
                with st.chat_message(name):
                    st.write(description)

        agent_system_messages = {
            name: generate_system_message(name, description, tools)
            for (name, tools), description in zip(
                names.items(), agent_descriptions.values()
            )
        }

        # for name, system_message in agent_system_messages.items():
        # st.write(name)
        # st.write(system_message)

        # @title topic_specifier_prompt
        topic_specifier_prompt = [
            SystemMessage(content="You can make a topic more specific."),
            HumanMessage(
                content=f"""{topic}

                    You are the moderator.
                    Please make the topic more specific.
                    Please reply with the specified quest in {word_limit} words or less.
                    Speak directly to the participants: {*names,}.
                    Do not add anything else."""
            ),
        ]

        specified_topic = llm(topic_specifier_prompt).content

        # st.write("Original topic:\n{topic}\n")
        # st.write("Detailed topic:\n{specified_topic}\n")

        agents = [
            DialogueAgentWithTools(
                name=name,
                system_message=SystemMessage(content=system_message),
                model=llm,
                tool_names=tools,
                top_k_results=2,
            )
            for (name, tools), system_message in zip(
                names.items(), agent_system_messages.values()
            )
        ]

        # @title 実行

        max_iters = min(2 * len(roles), 8)
        n = 0

        simulator = DialogueSimulator(
            agents=agents, selection_function=select_next_speaker
        )
        simulator.reset()
        simulator.inject("Moderator", specified_topic)
        with st.chat_message("Moderator"):
            st.write("(Moderator):" + specified_topic)

        while n < max_iters:
            name, message = simulator.step()
            with st.chat_message(name):
                st.write(
                    "("
                    + name
                    + "):"
                    + "  "
                    # + message
                    + translate_text(message)
                )

            n += 1
