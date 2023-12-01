"""
This is a Python script that serves as a frontend for a conversational AI model built with the `langchain` and `llms` libraries.
The code creates a web application using Streamlit, a Python library for building interactive web apps.
"""

# Import necessary libraries
import streamlit as st
from langchain.chains import ConversationChain
from streamlit_chat import message
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.conversation.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain import PromptTemplate

# Chrome Database - local vector database to store documents embedings
persist_directory = '/home/jarekmor/python_projects/Chat/db'
MODEL="gpt-3.5-turbo"

# Set Streamlit page configuration
st.set_page_config(page_title="🧠ChatBot🤖", layout="wide")
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
    input_text = st.text_input(
        "You: ",
        st.session_state["input"],
        key="input",
        placeholder="I am your AI assistant. Tell me your requirements ...",
        label_visibility="hidden",

    )
    return input_text

# Define function to start a new chat
def new_chat():
    """
    Clears session state and starts a new chat.
    """
    save = []
    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        save.append("User:" + st.session_state["past"][i])
        save.append("Bot:" + st.session_state["generated"][i])
    st.session_state["stored_session"].append(save)
    st.session_state["generated"] = []
    st.session_state["past"] = []
    st.session_state["input"] = ""
    st.session_state.entity_store = {}
    st.session_state.entity_memory.buffer.clear()
    st.session_state.stored_session = []
 
# Set up sidebar with various options
# with st.sidebar.expander(" 🛠️ Settings ", expanded=False):
#     # Option to preview memory store
#     # if st.checkbox("Preview memory store"):
#     #     st.write(st.session_state.entity_memory.entity_store)
#     # Option to preview memory buffer
#     if st.checkbox("Preview memory buffer"):
#         st.write(st.session_state.entity_memory.buffer)
#     MODEL = st.selectbox(
#         label="Model",
#         options=[
#             "gpt-3.5-turbo",
#             "text-davinci-003",
#             "text-davinci-002",
#             "code-davinci-002",
#         ],
#     )

# App
st.image("https://logosandtypes.com/wp-content/uploads/2022/07/openai.svg") # link or file name from local disk
st.header(" 🧠 :blue[AI ChatBot] 🤖")
st.title("Chat with your documents - pdf, doc documents retrival "  " :sunglasses:")
st.write("Do poprawnego działania programu Chat' potrzebujesz :blue[*OPENAI_API_KEY*] ")


# Ask the user to enter their OpenAI API key
API_O = st.sidebar.text_input(
    ":blue[Enter Your OPENAI API-KEY :]",
    placeholder="Paste your OpenAI API key here (sk-...)",
    type="password",
)

# Session state storage would be ideal
if API_O:

    # Create an OpenAI instance
    llm = ChatOpenAI(temperature=0, openai_api_key=API_O, model_name=MODEL, verbose=False)

    # Create a ConversationBufferMemory object if not already created
    if "entity_memory" not in st.session_state:
        st.session_state.entity_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    embeddings = OpenAIEmbeddings(openai_api_key=API_O)
    docsearch = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

    # Create the ConversationChain object with the specified configuration

    chain = ConversationalRetrievalChain.from_llm(
                    llm=llm, chain_type="stuff",
                    retriever=docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3}),
                    memory = st.session_state.entity_memory)
else:
    st.markdown(
        """ 
        ```
        - 1. Enter API Key + Hit enter 🔐 

        - 2. Ask anything via the text input widget

        Your API-key is not stored in any form by this app. However, for transparency ensure to delete your API once used.
        ```
        
        """
    )
    st.sidebar.warning(
        "API key required to try this app. The API key is not stored in any form."
    )
    st.sidebar.info("Your API-key is not stored in any form by this app. However, for transparency ensure to delete your API once used.")


# Add a button to start a new chat

if API_O:
    st.sidebar.button("New Chat", on_click=new_chat, type="primary")

# Get the user input
user_input = get_text()

# Generate the output using the ConversationChain object and the user input, and add the input/output to the session
if user_input:
    if API_O:
        
        template = """
        Your are helpful AI assistant who helps with finding informations which meets requirements provided in the following question.{question}
        """
        prompt = PromptTemplate(input_variables=["question"], template=template)
        
        output = chain({"question": user_input, "chat_history": st.session_state["generated"]}, prompt)
        
        st.session_state["past"].append(user_input)
        st.session_state["generated"].append(output["answer"])
               
    else:
        st.error("You need to enter your API key in the sidebar")

# Allow to download as well
download_str = []

# Display the conversation history using an expander, and allow the user to download it
with st.expander("Conversation", expanded=True):
    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        st.info(st.session_state["past"][i], icon="🧐")
        st.success(st.session_state["generated"][i], icon="🤖")
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

