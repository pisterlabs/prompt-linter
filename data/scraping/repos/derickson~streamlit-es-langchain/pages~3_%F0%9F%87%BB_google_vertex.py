import streamlit as st
import openai
from langchain.llms import OpenAI

from langchain.chat_models import ChatOpenAI
from langchain import ConversationChain
from langchain.prompts.prompt import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory

from resources import load_vertexai

## Initialize chat history
if "memory_google" not in st.session_state:
    st.session_state.memory_google =  ConversationBufferWindowMemory(ai_prefix="AI", k=5)

## Header with a clear memory button
col1, col2 = st.columns([3,1])
with col1:
    st.title('Google Vertex')
with col2:
    if st.button("Clear Memory"):
        st.session_state.memory_google.clear()

f"""
This is a basic example showing a chat loop with LangChain and Google PaLM2 on VertexAI 

"""

TEMPLATE = """You are helpful AI Assustant that answers questions.

Current conversation:
{history}
Human: {input}
AI:"""

## Create conversational LLM Chain
if "conversation_google" not in st.session_state:
    template = TEMPLATE
    PROMPT = PromptTemplate(input_variables=["history", "input"], template=template)
    st.session_state.conversation_google = ConversationChain(
        prompt=PROMPT,
        llm=load_vertexai(),
        verbose=True,
        memory=st.session_state.memory_google,
    )

## When a new human chat is received
def chatExchange(humanInput):
    return st.session_state.conversation_google.run(humanInput)

## The Chat input at the bottom of the screen
if prompt := st.chat_input("What is up?"):
    chatExchange(prompt)
    
# Display chat messages from history on app rerun
for message in st.session_state.memory_google.buffer_as_messages:
    with st.chat_message("user" if message.type == 'human' else 'assistant' ):
        st.markdown(message.content)
    
