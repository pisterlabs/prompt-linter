import numpy as np
import streamlit as st
import replicate
import os
import pandas as pd
import openai
from dotenv import load_dotenv
from langchain.agents import load_tools, initialize_agent, create_pandas_dataframe_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.tools import DuckDuckGoSearchRun
from langchain.chat_models import ChatOpenAI
load_dotenv()
#os.environ["OPENAI_API_KEY"] = st.secrets['OPENAI_API_KEY']
from langchain.llms import OpenAI
# App title
st.set_page_config(page_title="🦙💬 ChatVirgin")

# Replicate Credentials   
with st.sidebar:
    st.title('Eminem^50C')
     #'REPLICATE_API_TOKEN': ''' in st.secrets:
        #st.success('API key already provided!', icon='✅')
        #replicate_api = ['REPLICATE_API_TOKEN']''' #st.secret
    
    replicate_api = st.text_input('Enter Replicate API token:', type='password')
    if not (replicate_api.startswith('r8_') and len(replicate_api)==40):
        st.warning('Please enter your credentials!', icon='⚠️')
    else:
        st.success('Proceed to entering your prompt message!', icon='👉')
        os.environ['REPLICATE_API_TOKEN'] = replicate_api

    st.subheader('Models and parameters')
    selected_model = st.sidebar.selectbox('Choose a Llama2 model', ['phi-1.5', 'Llama2-7B'], key='selected_model')
    if selected_model == 'phi-1.5':
        llm = 'lucataco/phi-1.5:1503b791710440d857384e4d7057d9ebf645313ae8cce5c3f2b02585d910b3d0'
    elif selected_model == 'Llama2-7B':
        llm = '"meta/llama-2-7b-chat:8e6975e5ed6174911a6ff3d60540dfd4844201974602551e10e9e87ab143d81e"'
    temperature = st.sidebar.slider('temperature', min_value=0.01, max_value=5.0, value=0.1, step=0.01)
    top_p = st.sidebar.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
    max_length = st.sidebar.slider('max_length', min_value=32, max_value=128, value=120, step=8)

st.subheader('Models')
tab1, tab2, tab3 = st.tabs(["phi-1.5", "llama2-7b", "ChatGpt"])
with tab1:
    llm = 'lucataco/phi-1.5:1503b791710440d857384e4d7057d9ebf645313ae8cce5c3f2b02585d910b3d0'

with tab2:
    llm = 'meta/llama-2-7b-chat:8e6975e5ed6174911a6ff3d60540dfd4844201974602551e10e9e87ab143d81e'

with tab3:
    os.environ['OPENAI_API_KEY'] = "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"
    st.write("Upload a CSV file and enter a query to get an answer.")
    file =  st.file_uploader("Upload CSV file",type=["csv"])
    if not file:
        st.stop()
        data = pd.read_csv(file)
    with st.container():
        st.write("Data Preview:")
        #st.dataframe(data.head()) 
        df = pd.DataFrame(
            np.random.randn(50, 20),
            columns=('col %d' % i for i in range(20)))
        st.dataframe(df)
        agent = create_pandas_dataframe_agent(OpenAI(temperature=0.1),df,verbose=True) 

        query = st.text_input("Enter a query:") 

        if st.button("Execute"):
            answer = agent.run(query)
            st.write("Answer:")
            st.write(answer)
        
    llm = ChatOpenAI(model_name='gpt-3.5-turbo', openai_api_key = "OPENAI_API_KEY", streaming = True)
    openai_api_key = st.sidebar.text_input("OpenAI API Key", type = 'password')
    if 'messages' not in st.session_state:
        st.session_state['messages'] =[{'role': 'assistant', 'content':'Lets do this'}]
    for msg in st.session_state.messages:
        st.chat_message(msg['role']).write(msg['content'])
    if prompt := st.chat_input(placeholder = 'What would you suggest me to enrich my dataset for machine learning model training'):
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        st.chat_message('user').write(prompt)

        if not openai_api_key:
            st.info('Please add your OpenAI API key!')
            st.stop()
        llm = ChatOpenAI(model_name='gpt-3.5-turbo', openai_api_key = openai_api_key, streaming = True)

        search_agent = initialize_agent(
            tools = [DuckDuckGoSearchRun(name='Search')],
            llm = llm,
            agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            handle_parsing_errors = True,
        )
        with st.chat_message('assistan'):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts = False)
            response = search_agent.run(st.container(), expand_new_thoughts = False)
            st.session_state.messages.append({'role':'assistant', 'content':response})
            st.write(response)



st.subheader('Parameters')
with st.sidebar:    
    temperature = st.sidebar.slider('temperature', min_value=0.01, max_value=5.0, value=0.1, step=0.01)
    top_p = st.sidebar.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
    max_length = st.sidebar.slider('max_length', min_value=32, max_value=128, value=120, step=8)

    

   

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Lets Critique your data!"}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

# Function for generating LLaMA2 response. Refactored from https://github.com/a16z-infra/llama2-chatbot
def generate_llama2_response(prompt_input):
    string_dialogue = "You are dataset evaluater. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'."
    for dict_message in st.session_state.messages:
        if dict_message["role"] == "user":
            string_dialogue += "User: " + dict_message["content"] + "\n\n"
        else:
            string_dialogue += "Assistant: " + dict_message["content"] + "\n\n"
    output = replicate.run('lucataco/phi-1.5:1503b791710440d857384e4d7057d9ebf645313ae8cce5c3f2b02585d910b3d0', 
                           input={"prompt": f"{string_dialogue} {prompt_input} Assistant: ",
                                  "temperature":temperature, "top_p":top_p, "max_length":max_length, "repetition_penalty":1})
    return output

# User-provided prompt
if prompt := st.chat_input(disabled=not replicate_api):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message(name = "user", avatar="💬"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message(name = "assistant", avatar="👨‍💻"):
        with st.spinner("Thinking..."):
            response = generate_llama2_response(prompt)
            placeholder = st.empty()
            full_response = ''
            for item in response:
                full_response += item
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)




