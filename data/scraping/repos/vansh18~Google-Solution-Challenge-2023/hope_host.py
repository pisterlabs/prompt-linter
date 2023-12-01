import streamlit as st
from streamlit_chat import message

from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationEntityMemory
from langchain.chains.conversation.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from langchain.llms import OpenAI

from langchain import OpenAI, ConversationChain, LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory

import os
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

template = st.secrets["template"]

MODEL = 'text-davinci-003' 
K = 10 

st.set_page_config(page_title='HOPE', layout='wide')
st.title("HOPE")

if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "past" not in st.session_state:
    st.session_state["past"] = []
if "input" not in st.session_state:
    st.session_state["input"] = ""
if "stored_session" not in st.session_state:
    st.session_state["stored_session"] = []

def get_text():

    input_text = st.text_input("You: ", st.session_state["input"], key="input",
                            placeholder="I am your HOPE! Ask me anything ...", 
                            label_visibility='hidden')
    return input_text

def new_chat():
    save = []
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        save.append("User:" + st.session_state["past"][i])
        save.append("Bot:" + st.session_state["generated"][i])        
    st.session_state["stored_session"].append(save)
    st.session_state["generated"] = []
    st.session_state["past"] = []
    st.session_state["input"] = ""

llm = OpenAI(temperature=0)

if 'entity_memory' not in st.session_state:
        st.session_state.entity_memory = ConversationBufferWindowMemory(k=K)

if 'entity_memory' not in st.session_state:
        st.session_state.entity_memory = ConversationBufferWindowMemory(k=K)

prompt = PromptTemplate(
    input_variables=["history", "human_input"], 
    template=template
)

bot_chain = LLMChain(
    llm=OpenAI(temperature=0), 
    prompt=prompt, 
    verbose=True, 
    memory=st.session_state.entity_memory
)    

st.sidebar.button("New Chat", on_click = new_chat, type='primary')

user_input = get_text()

if user_input:
    output = bot_chain.predict(human_input = user_input)  
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)
    
with st.expander("Conversation", expanded=True):
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')

for i, sublist in enumerate(st.session_state.stored_session):
        with st.sidebar.expander(label= f"Conversation-Session:{i}"):
            st.write(sublist)

if st.session_state.stored_session:   
    if st.sidebar.checkbox("Clear-all"):
        del st.session_state.stored_session
