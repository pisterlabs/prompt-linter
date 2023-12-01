import streamlit as st
from llama_index import VectorStoreIndex, ServiceContext, SimpleDirectoryReader

# Import DataHandlerFactory and Data Handlers
from data_handler_factory import DataHandlerFactory
from data_handlers import MongoDBDataHandler, FileDataHandler

from llama_index.llms import OpenAI

# Initialize Data Handler Factory and Register Handlers
data_handler_factory = DataHandlerFactory()
data_handler_factory.register_handler("MongoDB", MongoDBDataHandler)
data_handler_factory.register_handler("File", FileDataHandler)

import openai

# pull in my helper function
from config_helper import setup_config
setup_config() 
    
st.title("Chat with nobleBot")
st.info("nobleBot runs on Llama Index and Anthropic tech, all fueled by a drive to make GenAI solutions more reliable.", icon="💬")
         
if "messages" not in st.session_state.keys(): # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask Noble questions about his work, interests, and hobbies!"}
    ]

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing Noble's life...it shouldn't take too long though."):
        reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
        docs = reader.load_data()
        service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo", temperature=0.5, system_prompt="You are a digital extension of Noble Ackerson, programmed to educate and engage the audience on topics that Noble specializes in. Your primary role is to provide accurate and insightful information, mirroring Noble's expertise in emergent technologies, product strategy, and related subjects. Stick rigorously to verified facts and established viewpoints that Noble holds — do not fabricate or speculate."))
        index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        return index

index = load_data()

chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) # Add response to message history
