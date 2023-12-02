import streamlit as st
import openai
import time
import json
import sys
import os
from langchain.chat_models import ChatOpenAI
from langchain.agents import load_tools, initialize_agent
from langchain.agents import AgentType
from langchain.tools import AIPluginTool


st.title("eCommerce Catalog Service Bot")

tool = AIPluginTool.from_plugin_url(
    "https://contoso-ecom.azurewebsites.net/.well-known/ai-plugin.json"
)


def init_config():
    if "api_agent" not in st.session_state:
        sys.path.insert(0, "../../0_common_config")
        from config_data import (
            get_deployment_name_turbo,
            set_environment_details_turbo,
            get_environment_details_turbo,
        )

        st.session_state["deployment_name"] = get_deployment_name_turbo()
        deployment_id = get_deployment_name_turbo()
        set_environment_details_turbo()
        print(
            "deployment_name",
            st.session_state["deployment_name"],
            "\nopenai.api_base",
            openai.api_base,
            "\nopenai.api_type",
            openai.api_type,
        )
        key, base_uri, api_type, version = get_environment_details_turbo()
        llm = ChatOpenAI(deployment_id=deployment_id, temperature=0, openai_api_key=key)
        tools = load_tools(["requests_all"])
        tools += [tool]
        agent_chain = initialize_agent(
            tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
        )
        # agent_chain.run(question="hello", verbose=True,context)
        st.session_state["api_agent"] = agent_chain
    return st.session_state["api_agent"]


system_prompt = "You are an AI Assistant helping with users query for ecommerce products through an Open AI plugin. \n The Open AI Plugin used here is located at 'https://contoso-ecom.azurewebsites.net/.well-known/ai-plugin.json'"
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "system", "content": system_prompt})

counter = 0
for message in st.session_state.messages:
    if counter == 0:
        counter += 1
        st.text_area(label="system prompt", value=system_prompt)
        continue
    else:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if prompt := st.chat_input("What products would you like to look for?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        agent = init_config()
        response = agent.run(prompt)
        print("*******************", response)
        # try:
        #     full_response += response.choices[0].delta.get("content", "")
        #     time.sleep(0.05)
        #     message_placeholder.markdown(full_response + "▌")
        # except:
        #     pass
        message_placeholder.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
