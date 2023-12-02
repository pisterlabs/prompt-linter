import os
from transformers import pipeline

from langchain.llms import HuggingFaceHub
from langchain.prompts import (PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, 
                               AIMessagePromptTemplate, ChatPromptTemplate)
from langchain.chains import LLMChain, ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.tools import wolfram_alpha
from langchain.agents import load_tools, initialize_agent

import streamlit as st
from utils import build_llm, compute_cost, completion_smart_goal, memory_to_pandas

# TODO: Reorganize and structure code
# TODO: Add text input area in the side for system prompt testing
# TODO: Include chat history in the model input
# TODO: Modularize LLM model selection
# TODO: Separate completion and beta chat goal setting tabs

# load memory globally
memory_path = "test_long_term_memory.json"
memory_df = memory_to_pandas(memory_path)

smart_gen = memory_df['AI_profiles'][5]

# --------- Local Utils --------- #
def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

# Function for generating LLM response
def generate_response(system_prompt, message_history):
    '''
    Function to set smart goal conversationally with LLM.
    params:
        system_prompt: dict
        message_history: str
    return:
        smart_goal: str
    '''
    # build chat llm
    chatgpt = build_llm(provider='ChatGPT4', max_tokens=150, temperature=1)
    
    # design prompt
    system_template = SystemMessagePromptTemplate.from_template("{system_prompt}")
    human_template = HumanMessagePromptTemplate.from_template("{chat_history}")

    # create the list of messages
    chat_prompt = ChatPromptTemplate.from_messages([
        system_template,
        human_template
    ])
    
    # Create the complete prompt with conversation history
    chat_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in message_history])
    prompt = {
        'system_prompt': system_prompt,
        'chat_history': chat_history  
    }
    
    # Build chain
    chain = LLMChain(llm=chatgpt, prompt=chat_prompt)
    
    response = chain.run(prompt)  # Pass 'prompt' here
    
    cost = compute_cost(len(response.split()), 'gpt-4')
    
    return response, cost
# -------------------------------- #

st.set_page_config(page_title="🤗💬 SmartBot Test")


with st.sidebar:
    st.title('🤗💬 SmartBot')
    st.subheader('Powered by 🤗 Language Models')
    system_prompt = st.text_area("Enter your system prompt here", height=150)
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)
            
# Store LLM generated responses
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
if "total_cost" not in st.session_state:
    st.session_state.total_cost = 0

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


# User-provided prompt
# We isntantiate a new prompt with each chat input
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response, cost = generate_response(system_prompt, st.session_state.messages) 
            st.write(response)
            st.session_state.total_cost += cost
            st.sidebar.write(f"Cost of interaction: {cost}")
            st.sidebar.write(f"Total cost: {st.session_state.total_cost}")
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)
