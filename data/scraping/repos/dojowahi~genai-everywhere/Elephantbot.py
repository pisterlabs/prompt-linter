# Import necessary libraries
import streamlit as st
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationEntityMemory
from langchain.chains.conversation.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from langchain.llms import VertexAI
from langchain.chat_models import ChatVertexAI
from langchain.embeddings import VertexAIEmbeddings
from langchain.schema import AIMessage, HumanMessage, SystemMessage
import streamlit as st
from helpers.chathelper import get_text, new_chat
from helpers.vidhelper import streamlit_hide

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set Streamlit page configuration
st.set_page_config(page_title="Elephant Bot", layout="centered")
# Initialize session states
if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "past" not in st.session_state:
    st.session_state["past"] = []
if "input" not in st.session_state:
    st.session_state["input"] = ""
if "stored_session" not in st.session_state:
    st.session_state["stored_session"] = []

streamlit_hide()
# Set up sidebar with various options
# with st.sidebar.expander("Settings ", expanded=False):
with st.sidebar:
    if st.checkbox("Memory buffer"):
        st.write(st.session_state.entity_memory.buffer)

    K = st.number_input(
        " (#)Summary of prompts to consider", min_value=3, max_value=1000
    )

# Set up the Streamlit app layout
st.title("Elephant Bot :elephant:")
st.markdown(
    """ 
        > :black[**A Chatbot that remembers everything**]
        """
)

llm = ChatVertexAI(temperature=0.5, max_output_tokens=128)

# Create a ConversationEntityMemory object if not already created
if "entity_memory" not in st.session_state:
    st.session_state.entity_memory = ConversationEntityMemory(llm=llm, k=K)

# Create the ConversationChain object with the specified configuration
Conversation = ConversationChain(
    llm=llm,
    prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE,
    memory=st.session_state.entity_memory,
)


# Add a button to start a new chat
st.sidebar.button("New Chat", on_click=new_chat, type="primary")

# Get the user input
user_input = get_text()
logger.info(f"E-Bot User input:{user_input}")

# Generate the output using the ConversationChain object and the user input, and add the input/output to the session
if user_input:
    with st.spinner("Thinking..."):
        output = Conversation.run(input=user_input)
        logger.info(f"E-Bot Output:{output}")

        st.session_state.past.append(user_input)
        st.session_state.generated.append(output)

# Allow to download as well
download_str = []
# Display the conversation history using an expander, and allow the user to download it
with st.expander("Conversation", expanded=True):
    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        st.info(st.session_state["past"][i])
        st.success(st.session_state["generated"][i])
        download_str.append(st.session_state["past"][i])
        download_str.append(st.session_state["generated"][i])

    # Can throw error - requires fix
    download_str = "\n".join(download_str)
    if download_str:
        st.download_button("Download", download_str)

# Display stored conversation sessions in the sidebar
for i, sublist in enumerate(st.session_state.stored_session):
    with st.sidebar.expander(label=f"Conversation-Session:{i}"):
        st.write(sublist)

# Allow the user to clear all stored conversation sessions
if st.session_state.stored_session:
    if st.sidebar.checkbox("Clear-all"):
        del st.session_state.stored_session
