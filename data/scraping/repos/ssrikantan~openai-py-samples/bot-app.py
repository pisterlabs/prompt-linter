import streamlit as st
import openai
import time
import json
import sys


def init_config():
    if "deployment_name" not in st.session_state:
        sys.path.insert(0, "../../0_common_config")
        from config_data import (
            set_environment_details_gpt4,
            get_deployment_name_gpt4_32k,
        )

        st.session_state["deployment_name"] = get_deployment_name_gpt4_32k()
        set_environment_details_gpt4()
        print(
            "deployment_name",
            st.session_state["deployment_name"],
            "\nopenai.api_base",
            openai.api_base,
            "\nopenai.api_type",
            openai.api_type,
        )
    return st.session_state["deployment_name"]


st.title("Employment Offer Generator")

doc_template_name = "1-EmploymentOfferTemplate.txt"

if "messages" not in st.session_state:
    st.session_state.messages = []
    system_prompt = ""
    with open("metaprompt-1.txt", "r") as file:
        system_prompt += file.read()

    with open(doc_template_name, "r") as file:
        system_prompt += "\n" + file.read()

    with open("metaprompt-2.txt", "r") as file:
        system_prompt += "\n" + file.read()

    st.session_state.messages.append({"role": "system", "content": system_prompt})
    st.text_area(label="System Prompt", value=system_prompt, height=500)

counter = 0
for message in st.session_state.messages:
    if counter == 0:
        counter += 1
        continue
    else:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if prompt := st.chat_input("Hello!!!"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        deployment_name = init_config()
        for response in openai.ChatCompletion.create(
            engine=deployment_name,
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            temperature=0,
            stream=True,
        ):
            try:
                full_response += response.choices[0].delta.get("content", "")
                time.sleep(0.05)
                message_placeholder.markdown(full_response + "▌")
            except:
                pass
        message_placeholder.markdown(full_response)
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )
