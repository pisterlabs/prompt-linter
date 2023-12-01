import streamlit as st
import pandas as pd
import openai
import base64
import os

from functions.show_data_info import show_data_info
from functions.improve_prompt import improve_prompt
from functions.run_prompts_app import run_prompts_app

from src.keboola_storage_api.connection import add_keboola_table_selection
from src.st_aggrid.st_aggrid import interactive_table

image_path = os.path.dirname(os.path.abspath(__file__))

def set_page_config():
    st.set_page_config(
        page_title="Kai PromptLab",
        page_icon=image_path+"/static/keboola.png",
        layout="wide"
        )

def display_logo():
    logo_image = image_path+"/static/keboola_logo.png"
    logo_html = f'<div style="display: flex; justify-content: flex-end;"><img src="data:image/png;base64,{base64.b64encode(open(logo_image, "rb").read()).decode()}" style="width: 150px; margin-left: -10px;"></div>'
    st.markdown(f"{logo_html}", unsafe_allow_html=True)

def set_api_key(): 
    OPENAI_API_KEY = st.sidebar.text_input('Enter your OpenAI API Key:',
        help= """
        You can get your own OpenAI API key by following these instructions:
        1. Go to https://platform.openai.com/account/api-keys.
        2. Click on the __+ Create new secret key__ button.
        3. Enter an identifier name (optional) and click on the __Create secret key__ button.
        """,
        type="password",
        )

    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    openai.api_key = OPENAI_API_KEY
    return OPENAI_API_KEY

def get_uploaded_file(upload_option):
    if upload_option == 'Connect to Keboola Storage':
        add_keboola_table_selection()
        st.session_state.setdefault('uploaded_file', None)
    #elif upload_option == 'Upload a CSV file':
    #    file = st.sidebar.file_uploader("Choose a file", type='csv')
    #    st.session_state['uploaded_file'] = file
    elif upload_option == 'Use Demo Dataset':
        file = image_path + "/data/sample_data.csv"
        st.session_state['uploaded_file'] = file
    return st.session_state.get('uploaded_file')

def display_main_content(uploaded_file, openai_api_key):
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        #st.sidebar.success("The table has been successfully uploaded.")
        
        show_data_info(df)
        if st.session_state['uploaded_file'] is not None:
            interactive_table()

        if not openai_api_key:
            st.warning("To continue, please enter your OpenAI API Key.")
            
        improve_prompt()
        run_prompts_app(df)
        st.text(" ")
        display_logo()
    else:
        st.markdown("""
        __Welcome to Kai PromptLab!__ 
                    
        🔄 Start by connecting to the Keboola storage, you'll need your API token to do this. Just go to _Settings_ in your Keboola account and find the _API Tokens_ tab (see the [documentation](https://help.keboola.com/management/project/tokens/) for more information).
        Once connected, you'll be able to select the bucket and table you want to work with. 
                            """)

    hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

def main():
    set_page_config()
    display_logo()
    st.title('Kai PromptLab 👩🏻‍🔬')

    openai_api_key = set_api_key()

    upload_option = st.sidebar.selectbox('Select an upload option:', 
                                    ['Connect to Keboola Storage',
                                    #'Upload a CSV file',
                                    'Use Demo Dataset'
                                     ], help="""
    You can get your own API token by following these instructions:
    1. Go to Settings in your Keboola account.
    2. Go to the __API Tokens__ tab.
    3. Click on __+ NEW TOKEN__ button, set it and __CREATE__.
    """)
    uploaded_file = get_uploaded_file(upload_option)

    display_main_content(uploaded_file, openai_api_key)
    
if __name__ == "__main__":
    main()