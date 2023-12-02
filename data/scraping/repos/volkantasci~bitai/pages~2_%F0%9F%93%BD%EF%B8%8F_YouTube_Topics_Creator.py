from os import environ

import streamlit as st
import requests
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
from api_config import API_HOST, API_PORT

from respond_beauty import make_it_beautiful


API_URLS = {
    "OpenAI ChatGPT-4": API_HOST + f":{API_PORT}" + "/api/v1/prediction/f69140b3-5ed6-4c6b-9f89-0968a6de70ea",
    "OpenAI GPT-3.5-Turbo Instruct": API_HOST + f":{API_PORT}" + "/api/v1/prediction/cb680c86-db57-4efc-b367-75f9d531c624",
    "Meta AI LLaMa 2 - 70b": API_HOST + f":{API_PORT}" + "/api/v1/prediction/a71bf91f-6250-4cc7-8c65-a783cbf2f5c3"
}

if "youtube_memory" not in st.session_state:
    st.session_state.youtube_memory = ConversationBufferMemory()

if "youtube_interface_html" not in st.session_state:
    st.session_state.youtube_interface_html = False


def handle_user_input(prompt):
    def query(payload):
        st.session_state.youtube_memory.chat_memory.add_user_message(prompt)
        selected_api_url = API_URLS[st.session_state.youtube_selected_model]
        response = requests.post(selected_api_url, json=payload)
        return response.json()

    with st.spinner("Generating topics for your video..."):
        output = query({
            "question": prompt,
        })

        st.session_state.youtube_memory.chat_memory.add_ai_message(output)


def main():
    if "youtube_selected_model" not in st.session_state:
        st.session_state.youtube_selected_model = "Meta AI LLaMa 2 - 70b"

    st.session_state.youtube_models = [
        "OpenAI ChatGPT-4",
        "OpenAI GPT-3.5-Turbo Instruct",
        "Meta AI LLaMa 2 - 70b"
    ]

    #  Add title and subtitle
    st.title(":orange[bit AI] 🤖")
    st.caption(
        "bitAI powered by these AI tools:"
        "OpenAI GPT-3.5-Turbo 🤖, HuggingFace 🤗, CodeLLaMa 🦙, Replicate and Streamlit of course."
    )
    st.subheader("Create Topics and Titles")

    with st.container():
        col1, col2 = st.columns(2)

        with col1:
            st.session_state.youtube_selected_model = st.selectbox("Select model to use:",
                                                                   st.session_state.youtube_models)

        with col2:
            st.write('<div style="height: 27px"></div>', unsafe_allow_html=True)
            second_col1, second_col2 = st.columns([2, 1])
            with second_col1:
                clear_button = st.button("🗑️ Clear history", use_container_width=True)
                if clear_button:
                    st.session_state.youtube_memory.clear()

            with second_col2:
                st.session_state.youtube_interface_html = st.toggle("HTML", value=False)

    prompt = st.chat_input("✏️ Enter video subject here you want to create topics for: ")
    if prompt:
        handle_user_input(prompt)

    st.sidebar.caption('<p style="text-align: center;">Made by volkantasci</p>', unsafe_allow_html=True)

    for message in st.session_state.youtube_memory.buffer_as_messages:
        if isinstance(message, HumanMessage):
            if st.session_state.youtube_interface_html:
                with open("templates/user_message_template.html") as user_message_template:
                    new_content = make_it_beautiful(message.content)
                    html = user_message_template.read()
                    st.write(html.format(new_content), unsafe_allow_html=True)
            else:
                st.chat_message("Human", avatar="🤗").write(message.content)
        elif isinstance(message, AIMessage):
            if st.session_state.youtube_interface_html:
                with open("templates/ai_message_template.html") as ai_message_template:
                    new_content = make_it_beautiful(message.content)
                    html = ai_message_template.read()
                    st.write(html.format(new_content), unsafe_allow_html=True)
            else:
                st.chat_message("AI", avatar="🤖").write(message.content)


if __name__ == "__main__":
    main()
