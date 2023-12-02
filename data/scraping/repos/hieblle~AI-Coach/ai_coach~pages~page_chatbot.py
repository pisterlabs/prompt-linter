import streamlit as st
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationEntityMemory
from langchain.chains.conversation.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from langchain.llms import OpenAI


# streamlit settings/page configuration
st.title("GPT-3 Chatbot")
st.sidebar.markdown("#  Chatbot Settings")

# initialize session states in streamlit

if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "past" not in st.session_state:
    st.session_state["past"] = []
if "input" not in st.session_state:
    st.session_state["input"] = ""
if "stored_session" not in st.session_state:
    st.session_state["stored_session"] = []

# get text input from user

def get_text():

    input_text = st.text_input("You: ", st.session_state["input"], key="input", placeholder="Your AI assistent here! Ask me anything!", label_visibility="hidden")

    return input_text

# start a new chat

def new_chat():
    """
    Clears session state and starts a new chat.
    """
    save = []
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        save.append("User:" + st.session_state["past"][i])
        save.append("Bot:" + st.session_state["generated"][i])        
    st.session_state["stored_session"].append(save)
    st.session_state["generated"] = []
    st.session_state["past"] = []
    st.session_state["input"] = ""
    st.session_state.entity_memory.store = {}
    st.session_state.entity_memory.buffer.clear()


# API key for OpenAI
API_0 = st.sidebar.text_input("API-Key", type="password")
model = st.sidebar.selectbox(label= 'Model', options=["text-davinci-003", "text-davinci-001", "text-ada-001"])

if API_0:
    # create OpenAI instance
    llm = OpenAI(
        temperature=0,
        openai_api_key=API_0,
        model_name = model,
        verbose=False
    )

    # create conversation memory

    if 'entity_memory' not in st.session_state:
        st.session_state.entity_memory = ConversationEntityMemory(llm=llm,k=10) 

    # create conversation chain

    Conversation = ConversationChain(
        llm=llm,
        prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE,
        memory= st.session_state.entity_memory
    )

else:
    st.error("No API-Key provided")

st.sidebar.button("New Chat", on_click=new_chat, type='primary')

# get user input
user_input = get_text()

# generate output
if user_input:
    output = Conversation.run(input=user_input)
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)
    
with st.expander("Conversation", expanded=True):
    for i in range(len(st.session_state['generated'])-1,-1,-1):
        st.info(st.session_state["past"][i])
        st.success(st.session_state["generated"][i])




# TODO: use definition of API key from main file, maybe save with OS module
# ebenso auf Journal Daten zugreifen + Chatbot als Therapeuten definieren, Ziele setzen. Oder wie in wisdomai mit Ken Wilber o.ä. trainieren.