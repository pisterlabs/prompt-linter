from langchain_experimental.llms.anthropic_functions import AnthropicFunctions
from langchain.chat_models import ChatAnthropic
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.callbacks import StreamlitCallbackHandler
import streamlit as st
from langchain_experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory


chat = ChatAnthropic(
    streaming=True,
    verbose=True,
    callback_manager=StreamlitCallbackHandler(st.container()),

)




if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    messages = [
    HumanMessage(
        content=prompt
    )
]
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        response = chat(messages, callbacks=[st_callback])
        st.write(response)