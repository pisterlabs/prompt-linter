import streamlit as st
import openai


def run_key_check(session_state):
    message_placeholder = st.empty()
    
    input_value = st.text_input('🔑 OpenAI API Key o Password', type='password', key="unique_input_key", placeholder="Escribe aquí:")
    
    if len(input_value) <= 10 and len(input_value) > 0:
        stored_password = st.secrets.get("PASSWORD")
        stored_openai_key = st.secrets.get("OPENAI_API_KEY")
        
        if input_value == stored_password:
            set_openai_key(session_state, stored_openai_key)
            return True
        else:
            message_placeholder.warning('Password incorrecto', icon="🔒")
    
    elif len(input_value) > 10:
        try:
            openai.api_key = input_value
            openai.Completion.create(engine="text-davinci-003", prompt="test", max_tokens=5)
            set_openai_key(session_state, input_value)
            return True
        except openai.error.AuthenticationError:
            message_placeholder.warning('Por favor, introduce una clave válida de OpenAI!', icon="⚠️")
    
    return False

def get_openai_key(session_state):
    if 'api_key' in session_state:
        return session_state['api_key']
    else:
        return None

def set_openai_key(session_state, api_key):
    session_state['api_key'] = api_key
