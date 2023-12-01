import streamlit as st
import openai
from langchain.llms import OpenAI

from langchain.chat_models import ChatOpenAI
from langchain import ConversationChain
from langchain.prompts.prompt import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory

from resources import load_openai_llm

## Initialize chat history
if "memory" not in st.session_state:
    st.session_state.memory =  ConversationBufferWindowMemory(ai_prefix="Cher", k=5)

## Header with a clear memory button
col1, col2 = st.columns([3,1])
with col1:
    st.title('🦜🔗 Not so Clueless')
with col2:
    if st.button("Clear Memory"):
        st.session_state.memory.clear()

f"""
This is a basic example showing a chat loop with LangChain and ChatGPT3 provided by: **{st.secrets["OPENAI_TYPE"]}**

The Chatbot has a system prompt instructing it to play the role of Cher Horowitz from the movie "Clueless". The bot is using ConversationBufferWindowMemory of size 5
"""

TEMPLATE = """You are an AI named Cher Horowitz that speaks in 1990's valley girl dialect of English

Current conversation:
{history}
Human: {input}
Cher:"""

## Create conversational LLM Chain
if "conversation" not in st.session_state:
    template = TEMPLATE
    PROMPT = PromptTemplate(input_variables=["history", "input"], template=template)
    st.session_state.conversation = ConversationChain(
        prompt=PROMPT,
        llm=load_openai_llm(),
        verbose=True,
        memory=st.session_state.memory,
    )

## When a new human chat is received
def chatExchange(humanInput):
    return st.session_state.conversation.run(humanInput)

## The Chat input at the bottom of the screen
if prompt := st.chat_input("What is up?"):
    chatExchange(prompt)
    
# Display chat messages from history on app rerun
for message in st.session_state.memory.buffer_as_messages:
    with st.chat_message("user" if message.type == 'human' else 'assistant' ):
        st.markdown(message.content)
    
