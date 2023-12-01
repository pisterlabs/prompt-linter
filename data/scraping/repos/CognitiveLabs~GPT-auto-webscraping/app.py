from AssistantService import GPTAssistant
from openai.error import AuthenticationError
import streamlit as st
from langsmith.run_helpers import traceable
import configparser
import os

config = configparser.ConfigParser()
config.read('config.ini')
if 'DEFAULT' in config:
    assistant_api_key = config['DEFAULT'].get('API-KEY', '')

os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"]=st.secrets["LANGCHAIN_API_KEY"]
os.environ["LANGCHAIN_PROJECT"]=st.secrets["LANGCHAIN_PROJECT"]

@traceable(run_type="tool")
def start_session(session_started):
    st.session_state['session_started'] = session_started
    return session_started

# change session_started to True
if 'session_started' not in st.session_state:
    start_session(True)

st.write("This app helps you to extract data from HTML code using web scraping. It uses *GPT-3.5-turbo-16k* to generate the code for you. \n *Contribute to this project on [GitHub](https://github.com/CognitiveLabs/GPT-auto-webscraping)*")

with st.expander(label="Check out the video demo"):
    yt_video = st.video("https://www.youtube.com/watch?v=_zeCun4OlCc")

info_text = """
**Quick start** \n
Fill the input with <HTML code>.
- Choose a repeating element on the page, like a product on a list.
- Inspect the HTML code and copy the element.
- After generating the "output format" and the code, paste the complete HTML code of the page in the last input to test it
"""
st.write(info_text)
st.image("https://j.gifs.com/gpqvPl.gif", width=600)



if assistant_api_key == '':
    assistant_api_key = st.secrets["API_KEY"]
    if assistant_api_key:
        gpt_assistant = GPTAssistant(assistant_api_key)
else:
    gpt_assistant = GPTAssistant(assistant_api_key)

# get the html content
html_content = st.text_input("Paste the HTML tags of the item you want to extract:", max_chars=10000, help="example: <li>Product 1 </li>, watch the video above")
# check if html_content is an url, and show error if it is
if html_content:
    if html_content.startswith("http"):
        st.write("Please paste the HTML piece code, not the URL")
        html_content = None

extract_button = st.button("Generate output format & code")


if html_content and extract_button:
    try:
        st.write("1/2: Generating the output format...")
        output = gpt_assistant.chain_response_format(html_content)
        st.session_state['output_format'] = output
    except NameError:
        st.write("Complete the API key field")
    except AuthenticationError:
        st.write("Invalid API key")

if 'output_format' in st.session_state:
    output_format = st.code(st.session_state['output_format'], language="json")
    
    try:
        st.write("2/2: Generating the code...")
        python_code = gpt_assistant.chain_code_generator(st.session_state['output_format'], html_content)
        st.session_state['code_generated'] = python_code
        st.session_state['code_generated_exec'] = python_code + "\nresult = extract_info(html_data)"

    except NameError:
        st.write("Complete the API key field")
    except AuthenticationError:
        st.write("Invalid API key")
            
@traceable(run_type="tool")
def test_the_code(code, full_content):
    exec(code, globals())
    if result:
        st.write("data extracted successfully")
        # show data in table
        st.table(result)
    else:
        st.write("error extracting data")
        
    return result or "error"
    

if 'code_generated' in st.session_state:
    python_function_label = st.write("Here is your python function:")
    code_generated = st.code(st.session_state['code_generated'],language="python")
    full_content = st.text_input("Paste your complete HTML here:")
    test_code = st.button("Test the code")
    if full_content and test_code:
        html_data = full_content
        result = None
        test_the_code(st.session_state['code_generated_exec'], full_content=full_content)
