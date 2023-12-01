from streamlit_chat import message
import streamlit as st
import openai
import os

def generate_gpt_chat(prompt,model='gpeta',max_tokens=4000,temperature=0.5):
    response = openai.Completion.create(
        engine=model,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens
    )
    print(response)
    return response.choices[0].text


def chat():
    model = st.text_input("select model",key='model')
    max_tokens = st.slider('max tokens',100,4000)
    temperature = st.slider('temperature',0.0,1.0)
    
    
    prompt_conf = st.checkbox('prompt configuration')
    if prompt_conf:
        st.success('prompt configuration saved')        
        if 'generated' not in st.session_state:
            st.session_state['generated'] = []
        if 'past' not in st.session_state:
            st.session_state['past'] = []
        user_input=st.text_input("You:",key='input')
        if user_input:
            # output=generate_gpt_chat(user_input)
            output=generate_gpt_chat(prompt=user_input,model=model,max_tokens=max_tokens,temperature=temperature)
            #store the output
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)
        if st.session_state['generated']:
            for i in range(len(st.session_state['generated'])-1, -1, -1):
                message(st.session_state["generated"][i], key=str(i))
                message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')     


with st.sidebar:
    provider = st.selectbox('select provider', ['NA','azure openai', 'openai'])
    if provider == 'azure openai':
        openai.api_type = "azure"
        openai.api_base = st.text_input('azure endpoint') 
        openai.api_version = "2022-12-01"
        openai.api_key = st.text_input('here you need the key for azure openai api',type='password')
    elif provider == 'openai':
        openai.api_key = st.text_input('here you need the key for openai api')
        model = st.text_input('here you need the key for openai api')
        openai.api_key = st.text_input('here you need the key for openai api')
    else:
        st.warning('select provider')
    starter = st.checkbox('load configuration')
    if starter:
        st.success('configuration saved')
if starter: 
    chat()