from langchain import LLMChain, PromptTemplate, ConversationChain
from langchain.llms import LlamaCpp
from langchain.memory import ConversationBufferWindowMemory
from langchain.callbacks.base import BaseCallbackHandler
import streamlit as st
import os

def prompt():
    template = """You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'.

    {history}
    Human: {input}
    AI:"""
    return PromptTemplate(template=template, input_variables=['history', 'input'])

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

@st.cache_resource
def init_llm_chain(model_path, temperature, top_p, max_length):
    llm = LlamaCpp(
        model_path=model_path,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_length
    )
    return ConversationChain(
        prompt=prompt(),
        llm=llm,
        memory=ConversationBufferWindowMemory(k=5),
        verbose=True
    )

# App
st.set_page_config(page_title="🦙💬 Llama 2 Chatbot")

with st.sidebar:
    st.title('🦙💬 Llama 2 Chatbot')

    st.subheader('Models and parameters')
    selected_model = st.sidebar.selectbox('Choose a Llama2 model', ['Llama2-7B-q4', 'Llama2-7B-q8', 'Llama2-13B'], key='selected_model')
    if selected_model == 'Llama2-7B-q4':
        llm = './models/llama-2-7b-chat.ggmlv3.q4_K_M.bin'
    elif selected_model == 'Llama2-7B-q8':
        llm = './models/llama-2-7b-chat.ggmlv3.q8_0.bin'
    else:
        llm = './models/llama-2-13b-chat.ggmlv3.q4_1.bin'
    
    temperature = st.sidebar.slider('temperature', min_value=0.01, max_value=1.0, value=0.1, step=0.01)
    top_p = st.sidebar.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
    max_length = st.sidebar.slider('max_length', min_value=64, max_value=2048, value=512, step=8)

    chain = init_llm_chain(llm, temperature, top_p, max_length)    

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
    chain.memory.clear()
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

# User-provided prompt
if human_input := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": human_input})
    with st.chat_message("user"):
        st.write(human_input)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            stream_handler = StreamHandler(st.empty())
            response = chain.predict(input=human_input, callbacks=[stream_handler])
            
    # memory.save_context({"input": human_input}, {"output": response})
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)