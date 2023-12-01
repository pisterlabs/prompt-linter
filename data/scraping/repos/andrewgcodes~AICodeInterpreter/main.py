# Import necessary libraries
import streamlit as st
import openai
import os
import re
import math

# These are probably unnecessary but I include them in case exec() needs them.
import numpy
import matplotlib.pyplot
import sklearn
import pandas
import scipy
import sklearn
import requests
import bs4

# Configure Streamlit's page settings such as the title, icon, layout and the sidebar state
st.set_page_config(
    page_title="AI Code Interpreter",
    page_icon="👾",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# CSS styles to hide menu, footer and header of the Streamlit app
hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        </style>
        """
# Apply the above styles to the Streamlit app
st.markdown(hide_menu_style, unsafe_allow_html=True)

# Initialize OpenAI API key using environment variable
openai.api_key = os.getenv('OPENAI_API_KEY')

# Display title on the Streamlit application
st.title('**** COMMODORE 64-AI ****')
# st.write("READY")
# Function to remove all the text before a hashtag in a given string
def remove_before_hashtag(text):
    return re.sub(r'^.*?(?=#)', '', text)

def count_lines(string):
    return string.count('\n') + 1  # This assumes there is no trailing new line

# Function to validate that the response starts with a hashtag
def validate_response(response):
    return response.lstrip().startswith('#')

if 'ai_code' not in st.session_state:
    st.session_state['ai_code'] = ''

#if 'command' not in st.session_state:
#    st.session_state['command'] = ''

# User input field for entering commands
command = st.text_area('READY', value="Reverse the string 'solidgoldmagikarp' and count how many vowels it has", max_chars = 2000)

# Get OpenAI API to generate code
if st.button("LOAD 'aicode.py'") or st.session_state['ai_code'] != '':
    
    if command != '' or st.session_state['ai_code'] == '':
        #st.session_state['command'] = command

        if st.session_state['ai_code'] == '':
            try:
                # Indicate that processing is taking place
                with st.spinner('PROCESSING...'):
                    # Send a request to the OpenAI API to generate Python code
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are a Python programming assistant who writes code that can be executed using the exec() command. Only return the requested Python code. Do not include any commentary, dialogue, or extra information. Just return the raw code by itself in the format necessary for using exec(). The code should start with a hashtag comment."},
                            {"role": "user", "content": command + "\nMake sure the code prints out the result by using st.write(result). Do not use any print statements; instead of print() use st.write() to display outputs. The first line of code should be a comment starting with a hashtag. Only return the code. Do not talk or say anything. Do not give notes. Do not repeat the prompt. Only provide the code in a format suitable for exec(). You may need to generate synthetic data. Do not use any placeholder files, such as data.csv. Do not attempt webscraping. Do not use beautifulsoup. Implement error handling with try and except blocks and detailed error messages in st.write() into the code. Your response must follow this format: start with a # comment and end with 'st.write(result)' or another suitable streamlit command for the data type."}
                        ],
                        temperature = 0.0
                    )
                # Extract the AI generated code from the response
                ai_code = response.choices[0].message['content']
                # Remove all text before the first hashtag in the AI code
                ai_code = remove_before_hashtag(ai_code)
    
                # Validate the AI code
                if not validate_response(ai_code):
                    st.error('The response from the AI was not in the expected format. Please try again.')
                    st.stop()
                st.session_state.ai_code = ai_code
            except openai.OpenAIError as e:
                # Handle any errors that occur while connecting to the OpenAI API
                st.error(f"Error connecting to the OpenAI API: {e}")
        #else:
        #    ai_code = st.session_state['ai_code']

        # Display the AI generated code
        st.subheader("Code Block")
        st.warning("You MUST hit Cmd+Enter in the textbox after making any changes! \nAlso, you'll need to refresh the page if you want to enter a new prompt and generate new code.")
        #st.code(ai_code)

        numLines = count_lines(st.session_state.ai_code)
        st.text_area("", st.session_state.ai_code, key='ai_code', height = math.ceil(numLines*26) + 25)

        #st.text_area("", ai_code, height = math.ceil(numLines*25) + 25)

    else:
        # Prompt the user to enter a command if the input field is empty
        st.write('Please enter a command.')

# Button for executing the code
if st.button("RUN"):
    if st.session_state['ai_code'] != '':
        try:
            # Display the result of executing the AI generated code
            st.subheader("Code Interpreter Output")
            exec(st.session_state.ai_code)  # Execute the code stored in Session State
            st.success("Executed Successfully")
        except Exception as e:
            # Handle any errors that occur during code execution
            st.error(f"Error during code execution: {e}")
    else:
        st.write("You need to generate code first.")