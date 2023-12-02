import os
import sys
import re
import streamlit as st
import streamlit_authenticator as stauth
from dotenv import load_dotenv
from io import BytesIO
from io import StringIO

from langchain.agents import create_csv_agent
from langchain.chat_models import ChatOpenAI

from modules.history import ChatHistory
from modules.layout import Layout
from modules.utils import Utilities
from modules.sidebar import Sidebar
from modules.chatbot import Chatbot

import pandas as pd
import yaml
from yaml.loader import SafeLoader

#To be able to update the changes made to modules in localhost,
#you can press the "r" key on the localhost page to refresh and reflect the changes made to the module files.
def reload_module(module_name):
    import importlib
    import sys
    if module_name in sys.modules:
        importlib.reload(sys.modules[module_name])
    return sys.modules[module_name]

chatbot_module = reload_module('modules.chatbot')
history_module = reload_module('modules.history')
layout_module = reload_module('modules.layout')
utils_module = reload_module('modules.utils')
sidebar_module = reload_module('modules.sidebar')

Chatbot = chatbot_module.Chatbot
ChatHistory = history_module.ChatHistory
Layout = layout_module.Layout
Utilities = utils_module.Utilities
Sidebar = sidebar_module.Sidebar

def init():
    load_dotenv()
    st.set_page_config(layout="wide", page_icon="💬", page_title="IPAgent")
    hide_streamlit_style = """
            <style>
            #MainMenu {visibility: visible;}
            footer {visibility: hidden;}
            /*.block-container {padding-top:0.5rem !important}*/
             </style>
            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

def main():
    # Initialize the app
    init()

    try:
        with open('./.creds.yaml') as file:
            config = yaml.load(file, Loader=SafeLoader)
        authenticator = stauth.Authenticate(
            config['credentials'],
            config['cookie']['name'],
            config['cookie']['key'],
            config['cookie']['expiry_days']
        )
        name, authentication_status, username = authenticator.login('', 'main')
    except:
        print('no .creds.yaml found so not using web auth for logins...')
        authentication_status = True
        authenticator = None

    if authentication_status:

        # Instantiate the main components
        layout, sidebar, utils = Layout(), Sidebar(), Utilities()

        layout.show_header()

        user_api_key = utils.load_api_key()

        if not user_api_key:
            layout.show_api_key_missing()
        else:
            os.environ["OPENAI_API_KEY"] = user_api_key

            inputfile = './parsed_data.csv'

            uploaded_file = open(inputfile, 'rb')
            uploaded_file.seek(0)
            #uploaded_file_content = uploaded_file.read()

            file_container = st.expander("Patent Context:")
            shows = pd.read_csv(inputfile)
            file_container.write(shows)

            # uploaded_file = utils.handle_upload()

            if uploaded_file:
                # Initialize chat history
                history = ChatHistory()

                # Configure the sidebar
                sidebar.show_options(uploaded_file,authenticator)

                try:
                    chatbot = utils.setup_chatbot(
                        uploaded_file, st.session_state["model"], st.session_state["temperature"], shows
                    )
                    st.session_state["chatbot"] = chatbot

                    if st.session_state["ready"]:
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
                                output = st.session_state["chatbot"].conversational_chat(user_input)
                                history.append("assistant", output)

                        history.generate_messages(response_container)

                        # launch CSV Agent if button clicked
                        if st.session_state["show_csv_agent"]:

                            query = st.text_input(label="Use agent for precise information about the structure of your context", 
                                                placeholder="e-g : how many rows in my context ?"
                                                )
                            if query != "":

                                # format the CSV file for the agent
                                #print('running')
                                #uploaded_file_content = uploaded_file.read() # BytesIO(uploaded_file.getvalue())
                                #print(uploaded_file_content)

                                old_stdout = sys.stdout
                                sys.stdout = captured_output = StringIO()

                                # Create and run the CSV agent with the user's query
                                agent = create_csv_agent(ChatOpenAI(temperature=0), inputfile, verbose=True, max_iterations=4)
                                result = agent.run(query)

                                sys.stdout = old_stdout

                                # Clean up the agent's thoughts to remove unwanted characters
                                thoughts = captured_output.getvalue()
                                cleaned_thoughts = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', thoughts)
                                cleaned_thoughts = re.sub(r'\[1m>', '', cleaned_thoughts)

                                # Display the agent's thoughts
                                with st.expander("Display the agent's thoughts"):
                                    st.write(cleaned_thoughts)
                                    Utilities.count_tokens_agent(agent, query)

                                st.write(result)


                except Exception as e:
                    st.error(f"Error: {str(e)}")

        # sidebar.about()

    elif authentication_status is False:
        st.error('Username/password is incorrect')
    elif authentication_status is None:
        st.warning('Please enter your username and password')


if __name__ == "__main__":
    main()