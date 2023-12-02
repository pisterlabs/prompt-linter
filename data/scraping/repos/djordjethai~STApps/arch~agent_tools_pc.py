# This code is the is the chatbot OpenAI gpt-3.5-turbo model that uses embeddings it uses langchain as workflow,
# serp as google search tool, and pinecone as embedding index database all usint streamplit for web UI

from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
import openai
import time
import os
from langchain.agents import Tool
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.chains import RetrievalQA
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.utilities.google_search import GoogleSearchAPIWrapper
import streamlit as st
from langchain.memory import ConversationBufferMemory
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.callbacks.base import BaseCallbackHandler


st.subheader("AI Asistent je povezan na internet i na LangChain biblioteku")
col1, col2, col3 = st.columns(3)
# Hide footer
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        token = token.replace('"', '').replace(
            '{', '').replace('}', '').replace('_', ' ')
        self.text += token

        self.container.success(self.text)


# Initialize session states
if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "past" not in st.session_state:
    st.session_state["past"] = []
if "input" not in st.session_state:
    st.session_state["input"] = ""
if "stored_session" not in st.session_state:
    st.session_state["stored_session"] = []
if "messages" not in st.session_state:
    st.session_state["messages"] = []
    # Function to read a file and return its contents


def open_file(filepath):
    with open(filepath, "r", encoding="utf-8") as infile:
        sadrzaj = infile.read()
        infile.close()
        return sadrzaj

# Define function to get user input


def get_text():
    """
    Get the user input text.

    Returns:
        (str): The text entered by the user
    """
    input_text = st.text_input("You: ", st.session_state["input"], key="input",
                               placeholder="Your AI assistant here! Ask me anything ...",
                               label_visibility='hidden')
    return input_text

# Define function to start a new chat


def whoimi(input=""):
    """Positive AI asistent"""
    return ("Positive doo razvija AI Asistenta.")


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
    st.session_state.entity_mem = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True)
    st.session_state.entity_mem.clear()


# login procedure
with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)
name, authentication_status, username = authenticator.login('Login', 'main')

if st.session_state["authentication_status"]:
    # if login success enter the main program

    with col1:
        authenticator.logout('Logout', 'main', key='unique_key')
    # Read OpenAI API key from env
    openai.api_key = os.environ.get('OPENAI_API_KEY')

    # Retrieving API keys from env
    PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
    PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')

    # Retrieving API keys from env
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
    GOOGLE_CSE_ID = os.environ.get("GOOGLE_CSE_ID")
    # Initializing OpenAI and Pinecone APIs
    embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)
    pinecone.init(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_API_ENV
    )

    # Setting index name and printing a message to confirm that Pinecone has been initialized successfully
    index = pinecone.Index("embedings1")
    name_space = "koder"
    text_field = "text"

    vectorstore = Pinecone(
        index, embeddings.embed_query, text_field, name_space
    )

    if "entity_mem" not in st.session_state:
        st.session_state.entity_mem = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True)
    sistem = open_file("prompt_turbo.txt")
    odgovor = open_file("odgovor_turbo.txt")
    system_message_prompt = SystemMessagePromptTemplate.from_template(sistem)
    human_message_prompt = HumanMessagePromptTemplate.from_template("{text}")
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt])

    # prompt = ""

    # Creating a list of messages that includes a system message and an AI message
    # st.session_state.messages = [SystemMessage(content=sistem)]
    # st.session_state.system_message =  SystemMessage(content=sistem)
    # Loading the ChatOpenAI model and creating a question-answering chain
    memory = st.session_state.entity_mem
    placeholder = st.empty()
    # memorija je problematicna ako se koristi za razlicite teme!

    chat_box = st.empty()
    with chat_box:
        st.markdown("Streaming Chain of Thought and the Final Answer")
        stream_handler = StreamHandler(chat_box)

    chat = ChatOpenAI(
        openai_api_key=openai.api_key,
        temperature=0,
        model='gpt-3.5-turbo',
        streaming=True,
        callbacks=[stream_handler],

    )

    upit = " "
    # initializing tools Pinecone lookup and Intermediate Answer
    qa = RetrievalQA.from_chain_type(
        llm=chat,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )

    vectorstore = Pinecone(
        index, embeddings.embed_query, upit, name_space
    )

    # initializing tools internet search
    search = GoogleSearchAPIWrapper()

    # initialize agent tools
    tools = [
        Tool(
            name="search",
            func=search.run,
            description="Google search tool. Useful when you need to answer questions about recent events."
        ),
        Tool(
            name="Pinecone lookup",
            func=qa.run,
            verbose=True,
            description="Useful for when you need to answer questions about langchain and streamlit",
            return_direct=True
        ),
        Tool(
            name="Positive AI asistent",
            func=whoimi,
            description="Useful for when you need to answer questions about Positive AI asistent. Input should be Positive AI asistent "
        ),
    ]

    agent_chain = initialize_agent(tools, chat, agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
                                   messages=chat_prompt, verbose=True, memory=memory, handle_parsing_errors=True, max_iterations=4)

    # Create a Memory object if not already created
    if 'entity_mem' not in st.session_state:
        st.session_state.entity_mem = st.session_state.memory

    # Add a button to start a new chat
    with col2:
        st.button("New Chat", on_click=new_chat)
    download_str = []
    # container=st.container()

    st.session_state['question'] = ''

    # run conversation
    # This code creates a form using the placeholder.form function from Streamlit. Inside the form, there is a text input widget (st.text_input) where the user can input a question. The value of this widget is set to the value of the question key in the st.session_state dictionary.
    # The chat_prompt.format_prompt function is used to format the user's input into a prompt that can be understood by the LangChain AI model. The resulting prompt is stored in the pitanje variable.
    # When the user clicks the submit button, the st.session_state['question'] value is set to an empty string. The agent_chain.run function is called with the pitanje prompt as input, and the resulting answer is stored in the odgovor variable.
    # The past and generated lists in the st.session_state dictionary are updated with the user's input and the AI's response, respectively. These lists are then used to display the conversation history in a Streamlit expander widget.
    # Finally, the conversation history is joined into a string and stored in the download_str variable for potential download.

    with placeholder.form(key='my_form', clear_on_submit=True):
        upit = st.text_input(label='Postavite pitanje',
                             key='1', value=st.session_state['question'])
        formatted_prompt = chat_prompt.format_prompt(text=upit).to_messages()
        pitanje = formatted_prompt[0].content+formatted_prompt[1].content
        submit_button = st.form_submit_button(label='Submit')
        if submit_button:
            st.session_state['question'] = ''

            with st.spinner('Sacekajte trenutak...'):

                odgovor = agent_chain.run(pitanje)
                st.session_state.past.append(upit)
                st.session_state.generated.append(odgovor)
                time.sleep(3)
                chat_box.empty()
                with st.expander("Conversation", expanded=True):
                    for i in range(len(st.session_state['generated'])-1, -1, -1):

                        st.info(st.session_state["past"][i], icon="🧐")
                        st.success(st.session_state["generated"][i], icon="🤖")

                        download_str.append(st.session_state["past"][i])
                        download_str.append(st.session_state["generated"][i])

                    download_str = '\n'.join(download_str)

    if download_str:
        with col3:
            st.download_button('Download', download_str)

elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')
elif st.session_state["authentication_status"] is None:
    st.warning('Please enter your username and password')


