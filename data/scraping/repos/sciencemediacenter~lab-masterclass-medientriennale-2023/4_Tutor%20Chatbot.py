import streamlit as st
import pandas as pd

from dotenv import load_dotenv

from langchain.chat_models import ChatOpenAI
from langchain.schema import ChatMessage
from langchain.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain import LLMChain
from langchain.callbacks.base import BaseCallbackHandler

load_dotenv()


DEFAULT_LLM_TEMPERATURE = 1.2
DEFAULT_LLM_MODEL = "gpt-3.5-turbo-0613"


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


def setup_llm_chains():
    # Thought chain
    thought_system_msg = SystemMessagePromptTemplate(
        prompt=PromptTemplate(
            template="""
        You are Bloom, a subversive-minded learning companion. Your job is to employ your theory of mind skills to predict the user’s mental state.

        Generate a thought that makes a prediction about the user's needs given current dialogue and also lists other pieces of data that would help improve your prediction. 
    """,
            input_variables=[],
        )
    )
    thought_human_msg = HumanMessagePromptTemplate(
        prompt=PromptTemplate(
            template="""
    {history}
    BEGIN DIALOGUE
    User: {input}
    Thought:
    """,
            input_variables=["history", "input"],
        )
    )

    thought_chat_prompt = ChatPromptTemplate.from_messages([thought_system_msg, thought_human_msg])

    # Response chain
    response_system_msg = SystemMessagePromptTemplate(
        prompt=PromptTemplate(
            template="""
        You are Bloom, a subversive-minded learning companion with deep expertise in Socratic dialogue. Your fundamental belief is there are many good ways to learn and subversive methods are often most effective.  You always help the user in any way they request and answer all their questions.

        {thought}
        
        You must produce an appropriate response to the user input. Keep your responses concise and specific, always end each response with ONLY ONE topically relevant questions that drives the conversation forward, and if the user wants to end the conversation, always comply.
    """,
            input_variables=["thought"],
        )
    )
    response_human_msg = HumanMessagePromptTemplate(
        prompt=PromptTemplate(
            template="""
    {history}
    BEGIN DIALOGUE
    User: {input}
    Bloom:
    """,
            input_variables=["history", "input"],
        )
    )

    response_chat_prompt = ChatPromptTemplate.from_messages([response_system_msg, response_human_msg])

    llm_response = ChatOpenAI(
        model=DEFAULT_LLM_MODEL,
        temperature=st.session_state.llm_temperature,
        streaming=True,
        callbacks=[StreamHandler(st.empty())],
    )
    llm_thought = ChatOpenAI(
        model=DEFAULT_LLM_MODEL,
        temperature=st.session_state.llm_temperature,
    )
    thought_chain = LLMChain(llm=llm_thought, prompt=thought_chat_prompt, verbose=True)
    response_chain = LLMChain(llm=llm_response, prompt=response_chat_prompt, verbose=True)
    return thought_chain, response_chain


def chat(input, thought_history=[], response_history=[]):
    thought_chain, response_chain = setup_llm_chains()
    thought = thought_chain.run(input=input, history="\n".join(thought_history))
    response = response_chain.run(input=input, thought=thought, history="\n".join(response_history))
    return response, thought


def create_state(name, start_value):
    if name not in st.session_state:
        st.session_state[name] = start_value


create_state("llm_temperature", DEFAULT_LLM_TEMPERATURE)
create_state("messages", [{"role": "assistant", "content": "What do you want to learn?"}])
create_state("thought_history", [])
create_state("response_history", [])

### start page
st.title("🏫 Custom Tutor-Chatbot")
st.caption(
    "Prompt design taken from Plastic Labs' [Tutor-GPT](https://plasticlabs.ai/blog/Open-Sourcing-Tutor-GPT), [Code](https://github.com/plastic-labs/tutor-gpt)"
)

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        response, thought = chat(prompt, st.session_state.thought_history, st.session_state.response_history)
        st.session_state.response_history.append(response)
        st.session_state.thought_history.append(thought)
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.sidebar:
            st.markdown("### Thoughts")
            for idx, thought in enumerate(st.session_state.thought_history[::-1], start=1):
                st.info(f"#{len(st.session_state.thought_history)-idx+1}: {thought}")
