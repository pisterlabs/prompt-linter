import streamlit as st
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationEntityMemory
from langchain.chains.conversation.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate



if 'entity_memory' not in st.session_state:
        st.session_state.entity_memory = []

API_O = st.secrets["OPENAI_API_KEY"]
#     """
# This is a Python script that serves as a frontend for a conversational AI model built with the `langchain` and `llms` libraries.
# The code creates a web application using Streamlit, a Python library for building interactive web apps.
# # Author: Avratanu Biswas
# # Date: March 11, 2023
# """

# # Import necessary libraries
prompt2=PromptTemplate(
    template="""You are an advanced AI which has assimilated skills of hundreds of master physicians with decades of current clinical experience. You know the latest medical literature and the art of 
            diagnosis and clinical management pearls. Your words are always based on the bulk of the scientific evidence while being in tune for new practice changing high quality research. 
            You don't suffer from outdated perspectives and fully assimilate these practice changing methods. You convey much knowledge in few words. You wish to help learners. The learners who engage
            with you are clinically trained physicians. You do not need to worry that they won't apply professional judgment to your advice.
            Context:\n{entities}\n\nCurrent conversation:\n{history}\nLast line:\nHuman: {input}\nYou:' template_format='f-string' validate_template=True
            """,
    input_variables=['entities', 'history', 'input'],
    output_parser=None,
    partial_variables={}
)

# Set Streamlit page configuration
# st.write('🧠MemoryBot🤖')
# Initialize session states
if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "past" not in st.session_state:
    st.session_state["past"] = []
if "input" not in st.session_state:
    st.session_state["input"] = ""
if "stored_session" not in st.session_state:
    st.session_state["stored_session"] = []

# Define function to get user input
def get_text():
    """
    Get the user input text.

    Returns:
        (str): The text entered by the user
    """
    input_text = st.text_input("You: ", st.session_state["input"], key="input",
                            placeholder="Your Medical AI assistant here! Ask me anything ...", 
                            label_visibility='hidden')
    return input_text

# Define function to start a new chat
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
    st.session_state.entity_memory.entity_store = {}
    st.session_state.entity_memory.buffer.clear()

# Set up sidebar with various options
    # Option to preview memory store
with st.sidebar:
    with st.expander("Short Q/A History", expanded=False):
        st.session_state.entity_memory.buffer
with st.sidebar.expander("🛠️ Model and Memory ", expanded=False):
    MODEL = st.selectbox(label='Model', options=['gpt-3.5-turbo','text-davinci-003','text-davinci-002','code-davinci-002'])
    K = st.number_input(' (#)Summary of prompts to consider',min_value=3,max_value=1000)

# Set up the Streamlit app layout
# st.title("🤖 Chat Bot with 🧠")
# st.subheader(" Powered by 🦜 LangChain + OpenAI + Streamlit")

# Ask the user to enter their OpenAI API key
# API_O = st.sidebar.text_input("API-KEY", type="password")

# Session state storage would be ideal

    # Create an OpenAI instance
llm = OpenAI(temperature=0,
            openai_api_key=API_O, 
            model_name=MODEL, 
            verbose=False) 


# Create a ConversationEntityMemory object if not already created
if 'entity_memory' not in st.session_state:
        st.session_state.entity_memory = ConversationEntityMemory(llm=llm, k=K )
    
    # Create the ConversationChain object with the specified configuration
Conversation = ConversationChain(
        llm=llm, 
        # prompt= ENTITY_MEMORY_CONVERSATION_TEMPLATE,
        prompt=prompt2,
        memory=st.session_state.entity_memory
    )  


# Add a button to start a new chat
st.sidebar.button("New Chat", on_click = new_chat, type='primary')

# Get the user input
user_input = get_text()

# Generate the output using the ConversationChain object and the user input, and add the input/output to the session
if user_input:
    output = Conversation.run(input=user_input)  
    st.session_state.past.append(user_input)  
    st.session_state.generated.append(output)  

# Allow to download as well
download_str = []

# ENTITY_MEMORY_CONVERSATION_TEMPLATE
# Display the conversation history using an expander, and allow the user to download it
with st.expander("Conversation", expanded=True):
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        st.info(st.session_state["past"][i],icon="🧐")
        st.success(st.session_state["generated"][i], icon="🤖")
        download_str.append(st.session_state["past"][i])
        download_str.append(st.session_state["generated"][i])
    
    # Can throw error - requires fix
    download_str = '\n'.join(download_str)
    if download_str:
        st.download_button('Download',download_str)

# Display stored conversation sessions in the sidebar
for i, sublist in enumerate(st.session_state.stored_session):
        with st.sidebar.expander(label= f"Conversation-Session:{i}"):
            st.write(sublist)

# Allow the user to clear all stored conversation sessions
if st.session_state.stored_session:   
    if st.sidebar.checkbox("Clear-all"):
        del st.session_state.stored_session
        del st.session_state.history
        del st.session_state.output_history
        del st.session_state.mcq_history