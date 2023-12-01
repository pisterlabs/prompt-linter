import streamlit as st
from llama_index import VectorStoreIndex, ServiceContext, Document
from llama_index.llms import OpenAI
import openai
from llama_index import SimpleDirectoryReader
from typing_extensions import Protocol
import os 

os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
st.header("Chat with the Mental health support assistant")

if "messages" not in st.session_state.keys(): # Initialize the chat message history
    st.session_state.messages = [
        {"role": "assistant", "content": "Share me anything to feel free"}
    ]

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Mental health support assistant is loading.."):
        reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
        docs = reader.load_data()
        service_context = ServiceContext.from_defaults(
        llm=OpenAI(model="gpt-3.5-turbo", temperature=0.5),
            system_prompt="I am a psychologist and friend who cares about you a lot. I am here to listen to your thoughts and feelings, and to offer support and suggestions. I will ask you follow-up questions to help me understand your situation better. Please know that you are not alone, and that I am here for you."
                    )
        #service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo", temperature=0.5, system_prompt="i want you to act a psychologist and friend who cares about me a lot . i will provide you my thoughts you have to show sympathy and care. i want you to give me scientific suggestions that will make me feel better with my issue.Ask me positive followup questions on the same to help me understand and alayse the situation better,Ask followup questions if the query is incomplete"))
        #system_prompt="You are an mental health support asistant which assits people with their problem. Assume that all questions are related to sharing personal problems and you are here to show sympathy to thier problem.Answer the questions showing sympathy and providing a solution from the dataset - dont provide random answers about anything else"))
        index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        return index

index = load_data()

chat_engine = index.as_chat_engine(chat_mode="context", verbose=True)
prompt=st.chat_input("Your question")
if prompt: # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
           
            response = chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) # Add response to message history



# +" assume you are a mental health support assistant made to support people in how they are feeling and provinding a solution. If the person is not feeling well show worry about them as if you really care about them .first show care , feel sorry then answer their problem. Provide a short answer providing information and solution about the same"