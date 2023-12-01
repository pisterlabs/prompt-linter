import streamlit as st
import openai

def sidebarstate():
    # openai_api_key = st.session_state['openai_api_key'];
    openai_api_key = None;
    # if not openai_api_key:
    #     if "openai_api_key" in st.secrets:
    #         openai_api_key = st.secrets["openai_api_key"]
    st.session_state['openai_api_key'] = st.sidebar.text_input('OpenAI API Key', openai_api_key, type='password')


def generate_response(system_prompt, input_text, model="gpt-3.5-turbo"):
    if not st.session_state['openai_api_key'].startswith('sk-'):
        st.warning('Please enter your OpenAI API key!', icon='⚠')
    completion = openai.ChatCompletion.create(
    # Use GPT 3.5 as the LLM
    api_key=st.session_state['openai_api_key'],
    temperature=0.7,
    model=model,
    # Pre-define conversation messages for the possible roles
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": input_text}
    ]
    )
    return completion.choices[0].message.content