import json
from dotenv import load_dotenv
import os
import openai
import streamlit as st
from src.GPTInterface.SimpleQuery import SimpleQuery
from pathlib import Path
import sys
import extra_streamlit_components as stx
from openai.error import AuthenticationError, InvalidRequestError

# I don't really understand this, but it fixes import errors
sys.path.append(str(Path(__file__).resolve().parent.parent))

st.set_page_config(page_title="Smart-GPT", page_icon=":brain:", layout="centered", initial_sidebar_state="auto", menu_items=None)

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 


def get_manager():
    return stx.CookieManager()


cookie_manager = get_manager()


if 'error' not in st.session_state:
    st.session_state['error'] = ""



oaik = cookie_manager.get(cookie="OPENAI_API_KEY")
if oaik is None:
    st.error("No OpenAI API Key found. Please enter here:")
    key = st.text_input("OpenAI API Key", type="password")
    
    clicked = st.button("Save API Key")
    if clicked:
        cookie_manager.set("OPENAI_API_KEY", key)

if oaik is not None:
    clear = st.button("Clear saved API Key")
    if clear:
        cookie_manager.delete("OPENAI_API_KEY")
        st.session_state['error'] = ""


openai.api_key = oaik

input_text = st.text_area("Input prompt: ", height=200, max_chars=1000, key="input_prompt", disabled=oaik is None)  


if 'resolver_response' not in st.session_state:
    st.session_state['resolver_response'] = ''
    
if 'progress' not in st.session_state:
    st.session_state['progress'] = ""
    
if st.session_state['error'] != "":
    st.error(st.session_state['error'])

if 'details' not in st.session_state:
    st.session_state['details'] = dict()

def run_smart_gpt():
    try:
        # Add chain of thought suffix (Hebenstreit et al.) (https://arxiv.org/pdf/2305.02897.pdf)
        chain_of_thought_suffix = "Let's work this out in a step by step way to be sure we have the right answer:"
        prompt = input_text + "\n\n" + chain_of_thought_suffix

        query = SimpleQuery(quiet = True, model = model_choice)
        query.append("user", prompt)

        responses = []
        for i in range(num_of_alternatives):
            st.session_state['progress'] = f"Generating alternative {i + 1} of {num_of_alternatives}"
            my_bar.progress((i + 1) / num_of_alternatives, text=st.session_state['progress'])

            response = query.run()
            st.session_state['error'] = "Authentication error. Please check your OpenAI API key."
            responses.append(response)
        st.session_state['details']['responses'] = responses

        # Add Reflexion (Shinn, et al.) (https://arxiv.org/pdf/2303.11366.pdf)

        reflexion_prompt_suffix = f"You are a researcher tasked with investigating the {num_of_alternatives} response options provided. List the flaws and faulty logic of each answer option. Let's work this out in a step by step way to be sure we have all the errors:"

        reflexion_prompt = ""
        for i, response in enumerate(responses):
            reflexion_prompt += f"Answer number {i + 1}: \n{response}\n\n"

        reflexion_prompt += "\n\n" + reflexion_prompt_suffix

        query.append("user", reflexion_prompt)

        st.session_state['progress'] = "Generating reflexion"
        my_bar.progress(0.5, text=st.session_state['progress'])
        reflexion_response = query.run()
        query.append("assistant", reflexion_response)
        st.session_state['details']['reflexion'] = reflexion_response

        # Add Resolver (https://arxiv.org/pdf/2303.17071.pdf)
        resolver_prompt = f"You are a resolver tasked with:\n1) interpreting which of the {num_of_alternatives} answers the researcher thought was best\n2) Improving that answer based on the feedback and the content of other answers, and\n3) Printing the improved answer in full.\nLet's work this out in a step by step way to be sure we have the right answer:"

        query.append("user", resolver_prompt)
        
        st.session_state['progress'] = "Generating resolver response"
        my_bar.progress(0.75, text=st.session_state['progress'])
        resolver_resp = query.run()
        st.session_state["resolver_response"] = resolver_resp
        st.session_state['details']['resolver'] = resolver_resp
    
        st.session_state['progress'] = "Done!"
        my_bar.progress(1.0, text=st.session_state['progress'])
        st.session_state['error'] = ""
    except AuthenticationError:
        st.session_state['error'] = "Authentication error. Please check your OpenAI API key and try again."
        st.session_state['progress'] = ""
    except InvalidRequestError as e:
        if 'gpt-4' in str(e):
            st.session_state['error'] = "This API key does not have access to GPT-4 please select another model in the Options tab."
            st.session_state['progress'] = ""
        else:
            st.session_state['error'] = str(e)
            st.session_state['progress'] = ""
    except Exception as e:
        st.session_state['error'] = str(e)
        st.session_state['progress'] = ""


st.button("Send a message.", on_click=run_smart_gpt, type='primary', use_container_width=True, disabled= oaik is None)


m = st.markdown("""
<style>
div.stButton > button:first-child {
    box-shadow: 0px 0px 1em rgb(0 0 0 / 0.3);
}
</style>""", unsafe_allow_html=True)


my_bar = st.progress(0, text=st.session_state['progress'])
if st.session_state['progress'] == "":
    my_bar.empty()

st.markdown(st.session_state["resolver_response"])

with st.expander("Options"):
    model_choice = st.selectbox("Model", ["text-davinci-003", "gpt-3.5-turbo", "gpt-4"], index=1)
    num_of_alternatives = st.slider('How many alternatives to generate', 1, 5, 3)
    st.caption('Higher numbers of alternatives will take longer and cost more, but will result in higher quality final generations.\nReccomended 2-4.')
    
if st.session_state['progress'] == "Done!":
    with st.expander("Extra output info"):
        st.header("Responses")
        for i, response in enumerate(st.session_state['details']['responses']):
            st.subheader(f"Response {i + 1}")
            st.markdown(response)
        st.header("Reflexion")
        st.markdown(st.session_state['details']['reflexion'])
        st.header("Resolver")
        st.markdown(st.session_state['details']['resolver'])
