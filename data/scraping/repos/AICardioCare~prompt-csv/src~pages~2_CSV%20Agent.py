import re
import sys
from io import StringIO

import pandas as pd
import streamlit as st
from langchain.agents import create_pandas_dataframe_agent
from langchain.llms import OpenAI

from modules.history import ChatHistory
from modules.layout import Layout
from modules.sidebar import Sidebar
from modules.utils import Utilities


# To be able to update the changes made to modules in localhost (press r)
def reload_module(module_name):
    import importlib
    import sys

    if module_name in sys.modules:
        importlib.reload(sys.modules[module_name])
    return sys.modules[module_name]


layout_module = reload_module("modules.layout")
sidebar_module = reload_module("modules.sidebar")
utils_module = reload_module("modules.utils")

Sidebar = sidebar_module.Sidebar

st.set_page_config(layout="wide", page_icon="❤️", page_title="AI Cardio Care | Speak to your Heart💓")

# Instantiate the main components
layout, sidebar, utils = Layout(), Sidebar(), Utilities()

layout.show_header()

user_api_key = utils.load_api_key()
uploaded_file = utils.handle_upload(["csv"])

if not user_api_key:
    layout.show_api_key_missing()

# Configure the sidebar
sidebar.show_options()
sidebar.about()

if user_api_key and uploaded_file:
    uploaded_file.seek(0)

    # Read Data as Pandas
    data = pd.read_csv(uploaded_file)

    # Define pandas df agent - 0 ~ no creativity vs 1 ~ very creative
    chatbot = create_pandas_dataframe_agent(OpenAI(temperature=0, openai_api_key=user_api_key), data, verbose=True)

    # Initialize chat history
    history = ChatHistory()
    try:
        st.session_state["chatbot"] = chatbot

        # Create containers for chat responses and user prompts
        response_container, prompt_container = st.container(), st.container()

        with prompt_container:
            # Display the prompt form
            is_ready, user_input = layout.prompt_form()

            # Initialize the chat history
            history.initialize(uploaded_file)

            # Reset the chat history if button clicked
            if st.session_state["reset_chat"]:
                history.reset(uploaded_file)

            if is_ready:
                # Update the chat history and display the chat messages
                history.append("user", user_input)

                old_stdout = sys.stdout
                sys.stdout = captured_output = StringIO()

                output = st.session_state["chatbot"].run(user_input)

                sys.stdout = old_stdout

                history.append("assistant", output)

                # Clean up the agent's thoughts to remove unwanted characters
                thoughts = captured_output.getvalue()
                cleaned_thoughts = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", thoughts)
                cleaned_thoughts = re.sub(r"\[1m>", "", cleaned_thoughts)

                # Display the agent's thoughts
                with st.expander("Display the agent's thoughts"):
                    st.write(cleaned_thoughts)

        history.generate_messages(response_container)

    except Exception as e:
        st.error(f"Error: {str(e)}")
