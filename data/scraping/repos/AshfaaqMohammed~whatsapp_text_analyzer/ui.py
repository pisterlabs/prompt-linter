import streamlit as st
import openai

def set_state_if_absent(key, value):
    if key not in st.session_state:
        st.session_state[key] = value


def set_initial_state():
    set_state_if_absent('question', 'Ask Question')
    set_state_if_absent('result', None)
    set_state_if_absent('query_started', False)


def set_openai_api_key(api_key: str):
    st.session_state['OPENAI_API_KEY'] = api_key
    openai.api_key = api_key


def reset_results(*args):
    st.session_state.result = None
