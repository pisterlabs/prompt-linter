# Cohere's imports
import cohere as co
from conversant.prompt_chatbot import PromptChatbot
from qa.bot import GroundedQaBot

# Sreamlit
import streamlit as st
from streamlit_chat import message
import streamlit.components.v1 as components

# general imports
import os
import re
import logging
import pandas as pd
import datetime
import random
from typing import Literal, Optional, Union

# TODO: Modularize the program into different files
# TODO: add history memory for recent user inputs for questions

#------------------------------------------------------------

COHERE_API = st.sidebar.text_input("Cohere API key", value="", type="password") 
SERP_API =  st.sidebar.text_input("Google SERP API key", value="", type="password") 

# wait until user enters API keys
if COHERE_API == "" or SERP_API == "":
    st.error("Please enter your API keys to continue.")
    st.stop()

co = co.Client(COHERE_API)

#qa_bot = GroundedQaBot(COHERE_API, SERP_API)
health_e = PromptChatbot.from_persona("health-e", client=co)

#------------------------------------------------------------
COMPONENT_NAME = "streamlit_chat"

root_dir = os.path.dirname(os.path.abspath(__file__))
build_dir = os.path.join(root_dir, "frontend/build")

_streamlit_chat = components.declare_component(
    COMPONENT_NAME,
    path = build_dir)

# data type for avatar style
AvatarStyle = Literal[ 
    "adventurer", 
    "adventurer-neutral", 
    "avataaars",
    "big-ears",
    "big-ears-neutral",
    "big-smile",
    "bottts", 
    "croodles",
    "croodles-neutral",
    "female",
    "gridy",
    "human",
    "identicon",
    "initials",
    "jdenticon",
    "male",
    "micah",
    "miniavs",
    "pixel-art",
    "pixel-art-neutral",
    "personas",
]

# not a good approach. Must be replaced with a better way to check if the input is a question
def is_question(text):
    question_words = ["what", "when", "where", "who", "why", "how", "which", "?"]
    question_regex = "|".join(question_words)
    if re.search(f"^{question_regex}", text, re.IGNORECASE):
        return True
    else:
        return False

@st.cache(allow_output_mutation=True, show_spinner=True)
def query_bot(text_input: str,
              qa: bool):
    
    if qa:
        bot = GroundedQaBot(COHERE_API, SERP_API)
        output = bot.answer(text_input)

    else:
        output = health_e.reply(text_input)

    return output

def get_text():
    input_text = st.text_input("Hello! How may I assist you?", "", key="input_text")
    return input_text 

def chat_message_ui(message: str,
                    is_user: bool = False, # choose random string from AvatarStyle for default
                    avatar_style: Optional[AvatarStyle] = None,
                    seed: Optional[Union[int, str]] = 42,
                    key: Optional[str] = None):
    '''
    Streamlit chat frontend style and display
    '''
    if not avatar_style:
        avatar_style = "pixel-art-neutral" if is_user else "bottts"

    _streamlit_chat(message=message, seed=seed, isUser=is_user, avatarStyle=avatar_style, key=key)
    
def init_chat():
    '''
    Initialize all session states. 
    Streamlit requires unique keys per session state. 
    Keep all states separated and independent from each other to avoid multiprocessing issues.
    '''
    if ('patient_description' not in st.session_state):
        st.session_state['patient_description'] = []
    
    if ('healthE_output' not in st.session_state):
        st.session_state['healthE_output'] = []
        
    if ('patient_question' not in st.session_state):
        st.session_state['patient_question'] = []    
    
    if ("QA_output" not in st.session_state):
        st.session_state['QA_output'] = []
        
    if ("history_outputs" not in st.session_state):
        st.session_state['history_outputs'] = []
        
    if ("history_inputs" not in st.session_state):
        st.session_state['history_inputs'] = []
        
    if ("random_id" not in st.session_state):
        st.session_state['random_id'] = random.randint(0, 1000)
        
    if ("session_report" not in st.session_state):
        st.session_state['session_report'] = []

    logging.info(st.session_state.keys)
#---------------------------------------------------------------

st.title("Health-E: AI healthcare assistant")

st.markdown("This is a prototype of a healthcare assistant that is meant to help you find the right doctor and book appointments fast, accounting for the urgency of your injury. Now that we are sharing this with the world, if you have any feedback please reach out and contribute with a pull request. We would love to hear your thoughts!")

init_chat()

form = st.form(key="user_settings")
with form:
    st.write("Hi! I am Health-E, an AI nurse assistant. I can help connect you to the right doctor and book appointments fast based on the urgency of your injury. How may I assist you today?")
    
    # User input - Question or description
    user_input = st.text_input("Question or statement", key="intro_msg")
    logging.info("User input:" + user_input)

    # Submit button to answer
    generate_button = form.form_submit_button("Ask.")
    if generate_button:
        if user_input == "":
            st.error("Sure you don't wan't to ask anything?")
        
        if "?" in user_input:
            answer = query_bot(user_input, qa=True)
            logging.info("QA bot:" + answer)
            # append new question to user chat history
            st.session_state["patient_question"].append(user_input)
            st.session_state['history_inputs'].append(user_input)
            # append new qa bot output to bot chat history
            st.session_state['QA_output'].append(answer)
            st.session_state["history_outputs"].append(answer)
        else:
            # limited to one message memory
            if len(st.session_state["history_inputs"]) == 0:
                prompt = user_input
            else:
                prompt = st.session_state["history_inputs"][-1] + " " + user_input

            output = query_bot(prompt, qa=False)
            logging.info("Health-E bot:" + output)
            # append new description to user chat history
            st.session_state["patient_description"].append(user_input)
            st.session_state['history_inputs'].append(user_input)
            # append new health-e output to bot chat history
            st.session_state['healthE_output'].append(output)
            st.session_state["history_outputs"].append(output)

        
        logging.info("-------------------------------------------------------------------------")
        logging.info(" ")
        logging.info("------------------------------BOT OUTPUTS-------------------------------------------")
        logging.info(f"HealthE history: {st.session_state['healthE_output']}")
        logging.info(f"QA history: {st.session_state['QA_output']}")
        logging.info(f"Bot output history: {st.session_state['history_outputs']}")
        logging.info(" ")
        logging.info("------------------------------USER INPUTs-------------------------------------------")
        logging.info(f"User descriptions: {st.session_state['patient_description']}")
        logging.info(f"User questions: {st.session_state['patient_question']}")
        logging.info(f"User input history: {st.session_state['history_inputs']}")
        logging.info(" ")
        logging.info("-------------------------------------------------------------------------")


    if 'history_outputs' in st.session_state:

        first_chat = False
        qa = False

        for i in range(0, len(st.session_state['history_outputs']), 1):
            
            chat_message_ui(st.session_state['history_inputs'][i], is_user=True, key=str(i) + '_user')
            
            # clean diagnose string
            if "Source" not in st.session_state['history_outputs'][i]:
                components_of_diagnose = st.session_state['history_outputs'][i].split(",")
                qa = False
            
            else:
                components_of_diagnose = st.session_state['history_outputs'][i].split("Source:")
                qa = True
            
            first_chat = True
            chat_message_ui(st.session_state['history_outputs'][i], key=str(i), avatar_style="female")
    
        if first_chat != False:
            
            if qa == True:
                answer, source = components_of_diagnose[0], components_of_diagnose[1]
                
                dict_report = {"ID": st.session_state['random_id'], 
                        "session start time": datetime.datetime.now(), # Record the individual chat input start time
                        "Answer": answer,
                        "Source": source,
                        "Notes": " "}
            else:  
                dict_report = {"ID": st.session_state['random_id'], 
                        "session start time": datetime.datetime.now(), # Record the individual chat input start time
                        "Answer": " ",
                        "Source": " ",
                        "Notes": components_of_diagnose[-1]}
                  
                i=0
                while ":" in components_of_diagnose[i]:
                    key, value = components_of_diagnose[i].split(":")
                    dict_report[key] = value
                    i += 1
                
            st.session_state['session_report'].append(dict_report)
            logging.info(f"Session report: {st.session_state['session_report']}")
            
            # Display report/information identified so far by Health-E
            report_df = pd.DataFrame(st.session_state['session_report'])
            st.dataframe(st.session_state['session_report'])

if st.checkbox("Show available times"):
    #available_slots = ["Wednesday 11/23 5pm", "Saturday 11/26 9am", "Sunday 11/27 11am"]
    appointment_date = st.selectbox(
        f"I can book an appointment for you. I found the following slots available for the specialist you need:",
        ("Wednesday 11/23 5pm", "Saturday 11/26 9am", "Sunday 11/27 11am"))

    contact_option = st.selectbox(
        'How would you like to be contacted?',
        ('Email', 'Home phone', 'Mobile phone'))

    if st.button("Book appointment"):
        st.success(f"Your appointment has been succesfully booked for {appointment_date} and you will be contacted by {contact_option}", icon="✅")
        st.balloons()

        








































# # Cohere's imports
# import cohere as co
# from cohere.classify import Example
# from conversant import PromptChatbot
# from conversant.prompts import ChatPrompt

# # Sreamlit
# import streamlit as st

# # General imports
# import pandas as pd
# import textwrap
# import numpy as np
# import json

# import os

# os.environ["COHERE_API"] = "mhsnOPXxi1m91vlrQJ6VsKFoDVhiqlKPeYHtEsZV"

# COHERE_API = os.environ["COHERE_API"]

# co = co.Client(COHERE_API)

# # Let's let the user pick from the default persona

# st.title("Health-E AI bot")