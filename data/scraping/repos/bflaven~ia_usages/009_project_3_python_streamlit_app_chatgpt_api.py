#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
[env]
# Conda Environment
# NO CONDA ENV

conda create --name ai_chatgpt_prompts python=3.9.13
conda info --envs
source activate ai_chatgpt_prompts
conda deactivate
# if needed to remove
conda env remove -n [NAME_OF_THE_CONDA_ENVIRONMENT]


# update conda 
conda update -n base -c defaults conda

# to export requirements
pip freeze > ai_chatgpt_prompts.txt


# to install
pip install -r ai_chatgpt_prompts.txt

[path]
cd /Users/brunoflaven/Documents/03_git/BlogArticlesExamples/ai_chatgpt_prompts/project_3_streamlit/

[file]
streamlit run 009_project_3_python_streamlit_app_chatgpt_api.py


https://stackoverflow.com/questions/65602056/how-to-set-and-access-environment-variables-in-python-file-for-streamlit-app

https://techcommunity.microsoft.com/t5/healthcare-and-life-sciences/integrating-azure-openai-with-streamlit-with-example-source-code/ba-p/3809006

https://github.com/ajitdash/pview/blob/main/explaincode.py


pip install openai
pip install streamlit
pip install python-dotenv

"""

# os + dotenv to manage the OpenAI platform's licence
import os as os
from dotenv import load_dotenv

# streamlit
import streamlit as st

# OpenAI platform
import openai

# personal configuration
import config_values.values_conf as conf

# Load environment variables from .env file
load_dotenv()
# Get API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# load the key
openai.api_key = OPENAI_API_KEY


### 1. VALUES ###
TEXT_TITLE_APP = conf.TEXT_TITLE_APP
TEXT_SUBHEADER_APP = conf.TEXT_SUBHEADER_APP
TEXT_WARNING = conf.TEXT_WARNING
TEXT_OUTPUT = conf.TEXT_OUTPUT

LABEL_EXPANDER = conf.LABEL_EXPANDER

# Usecase Title
MENU_SIDEBAR_USECASE_TITLE_OPTIONS = conf.MENU_SIDEBAR_USECASE_TITLE_OPTIONS

# Usecase Description
MENU_SIDEBAR_USECASE_DESCRIPTION_OPTIONS = conf.MENU_SIDEBAR_USECASE_DESCRIPTION_OPTIONS


TEXT_HELP_2 = conf.TEXT_HELP_2
GREAT_EXPENDER_TEXT_1 = conf.GREAT_EXPENDER_TEXT_1
GREAT_EXPENDER_TEXT_2 = conf.GREAT_EXPENDER_TEXT_2
GREAT_EXPENDER_TEXT_3 = conf.GREAT_EXPENDER_TEXT_3
TEXT_WARNING_REPORT = conf.TEXT_WARNING_REPORT
TEXT_WARNING_REPORT_HELP = conf.TEXT_WARNING_REPORT_HELP

TEXT_HELP_1 = conf.TEXT_HELP_1
TEXT_HELP_2 = conf.TEXT_HELP_2

### 2. USECASES VALUES ###

# CASE_6 :: 6. English to other languages
CASE_6_LANGUAGES_SELECTION = conf.CASE_6_LANGUAGES_SELECTION

TEXT_HELP_CASE_6_1 = conf.TEXT_HELP_CASE_6_1
TEXT_HELP_CASE_6_2 = conf.TEXT_HELP_CASE_6_2

# help for parameters
TEXT_HELP_TEMPERATURE_PARAMETER = conf.TEXT_HELP_TEMPERATURE_PARAMETER
TEXT_HELP_TOKENS_PARAMETER = conf.TEXT_HELP_TOKENS_PARAMETER


# CASE_3 :: 3. Summarize for a 2nd grader
TEXT_HELP_CASE_3_1 = conf.TEXT_HELP_CASE_3_1

# CASE_12 :: 12. Movie to Emoji (movie_to_emoji)
TEXT_HELP_CASE_12_1 = conf.TEXT_HELP_CASE_12_1

# CASE_21 :: 21. TL DR summarization
TEXT_HELP_CASE_21_1 = conf.TEXT_HELP_CASE_21_1

# CASE_21 :: 21. TL DR summarization
TEXT_HELP_CASE_21_1 = conf.TEXT_HELP_CASE_21_1

# CASE_22 :: 22. Python bug fixer
TEXT_HELP_CASE_22_1 = conf.TEXT_HELP_CASE_22_1

# CASE_22 :: 23. Spreadsheet creator
TEXT_HELP_CASE_23_1 = conf.TEXT_HELP_CASE_23_1

# CASE_39 :: 39. Notes to summary
TEXT_HELP_CASE_39_1 = conf.TEXT_HELP_CASE_39_1


# The code below is for the layout of the page
if "widen" not in st.session_state:
    layout = "centered"
else:
    layout = "wide" if st.session_state.widen else "centered"

st.set_page_config(
    layout=layout, page_title=TEXT_TITLE_APP, page_icon="🔌"
)


class CaseChatGptDrawMainAppBody:

    def __init__(self):
        
        # CASE_3
        # 3. Summarize for a 2nd grader
        # Main navigation app
        self.options = MENU_SIDEBAR_USECASE_TITLE_OPTIONS
        self.menu = st.sidebar.selectbox(
            "Menu options", self.options, help=TEXT_HELP_1)
        # CASE_3
        # 3. Summarize for a 2nd grader
        # INFOS
        self.summarize_info = "Summarize so it translates difficult text into simpler concepts."
        
        
        # CASE_10
        # 10. Classification
        self.summarize_info = "Summarize text by adding a \'tl dr: \' to the end of a text passage."

    def greatExpender(self, label, text):
        with st.expander(label):
            st.info(f"{text}")
        
    # Version sidebar
    def detectVersion(self):
        st.sidebar.markdown('* * *')
        st.sidebar.markdown('**VERSIONS**')
        st.sidebar.write("streamlit ::", st.__version__)

    # Main app design
    def doApp(self):

        st.header(f'{TEXT_TITLE_APP}')
        st.markdown(f'{TEXT_SUBHEADER_APP}')
        st.warning(f'{TEXT_WARNING}')

        selected_case = self.options.index(self.menu)
        
        for i, option in enumerate(self.options):
            if i == selected_case:
                st.markdown(f'{option}')
    
                self.greatExpender(
                    LABEL_EXPANDER, f"{MENU_SIDEBAR_USECASE_DESCRIPTION_OPTIONS[selected_case]}")
                
                

    # Do the app
    def run(self):
        self.doApp()
        self.detectVersion()


# Instance the class
app = CaseChatGptDrawMainAppBody()
app.run()



