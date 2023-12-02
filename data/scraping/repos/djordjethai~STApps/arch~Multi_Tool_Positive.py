# This code is the is the chatbot OpenAI gpt-3.5-turbo model that uses embeddings it uses langchain as workflow,
# serp as google search tool, and pinecone as embedding index database all usint streamplit for web UI

from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
import os
import time
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
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
import random
st.subheader(
    "AI Asistent je povezan na internet, LangChain i Streamlit")
st.info("Mozete birati model i temperaturu, a bice prikazan i streaming output. Moguc je i Download chata. Ako menjate temu, bolje je odabrati opciju New Chat")

# Hide footer
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# This code handles the streaming output of the model. It takes in a container to display the output in and an initial text
# to display. It overrides the on_llm_new_token function to add new tokens to the text and display them in the container.


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        token = token.replace('"', '').replace(
            '{', '').replace('}', '').replace('_', ' ')
        self.text += token
        self.container.success(self.text)

    def reset_text(self):
        self.text = ""

    def clear_text(self):
        self.container.empty()


# Initialize session states
if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "past" not in st.session_state:
    st.session_state["past"] = []
if "input" not in st.session_state:
    st.session_state["input"] = ""
if "messages" not in st.session_state:
    st.session_state["messages"] = []


def open_file(filepath):
    with open(filepath, "r", encoding="utf-8") as infile:
        sadrzaj = infile.read()
        infile.close()
        return sadrzaj


def whoimi(input=""):
    """Positive AI asistent"""
    return ("Positive doo razvija AI Asistenta. U ovaj odgovor mozemo dodati bilo koji tekst.")


def new_chat():
    st.session_state["generated"] = []
    st.session_state["past"] = []
    st.session_state["input"] = ""
    st.session_state.memory.clear()
    st.session_state["messages"] = []


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

    with st.sidebar:
        st.caption("Ver 18.07.23")
        authenticator.logout('Logout', 'main', key='unique_key')
        st.button("New Chat", on_click=new_chat)
    download_str = []
    if "open_api_key" not in st.session_state:
        # Retrieving API keys from env
        st.session_state.open_api_key = os.environ.get('OPENAI_API_KEY')
    # Read OpenAI API key from env
    if "PINECONE_API_KEY" not in st.session_state:
        # Retrieving API keys from env
        st.session_state.PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
    if "PINECONE_API_ENV" not in st.session_state:
        st.session_state.PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')
    if "GOOGLE_API_KEY" not in st.session_state:
        # Retrieving API keys from env
        st.session_state.GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
    if "GOOGLE_CSE_ID" not in st.session_state:
        st.session_state.GOOGLE_CSE_ID = os.environ.get("GOOGLE_CSE_ID")
        # Initializing OpenAI and Pinecone APIs
    if "embeddings" not in st.session_state:
        st.session_state.embeddings = OpenAIEmbeddings(
            openai_api_key=st.session_state.open_api_key)
        pinecone.init(
            api_key=st.session_state.PINECONE_API_KEY,
            environment=st.session_state.PINECONE_API_ENV
        )
    if "index" not in st.session_state:
        # Setting index name
        st.session_state.index = pinecone.Index("embedings1")
    if "name_space" not in st.session_state:
        st.session_state.name_space = "stilovi"
    if "text_field" not in st.session_state:
        st.session_state.text_field = "text"
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = Pinecone(
            st.session_state.index,
            st.session_state.embeddings.embed_query,
            st.session_state.text_field,
            st.session_state.name_space
        )
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True)
    if "sistem" not in st.session_state:
        st.session_state.sistem = open_file("prompt_turbo.txt")
    if "odgovor" not in st.session_state:
        st.session_state.odgovor = open_file("odgovor_turbo.txt")
    if "system_message_prompt" not in st.session_state:
        st.session_state.system_message_prompt = SystemMessagePromptTemplate.from_template(
            st.session_state.sistem)
    if "human_message_prompt" not in st.session_state:
        st.session_state.human_message_prompt = HumanMessagePromptTemplate.from_template(
            "{text}")
    if "chat_prompt" not in st.session_state:
        st.session_state.chat_prompt = ChatPromptTemplate.from_messages(
            [st.session_state.system_message_prompt, st.session_state.human_message_prompt])

    with st.sidebar:
        model = st.selectbox(
            "Odaberite model", ("gpt-3.5-turbo",
                                "gpt-3.5-turbo-16k",
                                "gpt-4")
        )
        temp = st.slider('Set temperature (0=strict, 1=creative)',
                         0.0, 1.0, step=0.1)

    placeholder = st.empty()

    # 1.  Using the session state variable, we can check if the stream handler is already in the session state,
    #    if not, we create it and add it to the session state.
    # 2.  We use the reset_text function of the stream handler to reset the text of the stream handler.

    pholder = st.empty()
    with pholder.container():
        if "stream_handler" not in st.session_state:
            st.session_state.stream_handler = StreamHandler(pholder)
    st.session_state.stream_handler.reset_text()
    if "chat" not in st.session_state:
        st.session_state.chat = ChatOpenAI(
            openai_api_key=st.session_state.open_api_key,
            temperature=temp,
            model=model,
            streaming=True,
            callbacks=[st.session_state.stream_handler]
        )
    upit = []

    # initializing tools Pinecone lookup and Intermediate Answer
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = Pinecone(
            st.session_state.index, st.session_state.embeddings.embed_query, upit, st.session_state.name_space
        )
    if "qa" not in st.session_state:
        st.session_state.qa = RetrievalQA.from_chain_type(
            llm=st.session_state.chat,
            chain_type="stuff",
            retriever=st.session_state.vectorstore.as_retriever()
        )
    if "search" not in st.session_state:
        # initializing tools internet search
        st.session_state.search = GoogleSearchAPIWrapper()
        # initialize agent tools
    if "tools" not in st.session_state:
        st.session_state.tools = [
            Tool(
                name="search",
                func=st.session_state.search.run,
                description="Google search tool. Useful when you need to answer questions about recent events."
            ),
            Tool(
                name="Pinecone lookup",
                func=st.session_state.qa.run,
                verbose=False,
                description="Useful for when you need to write an article using the style of a specific person. Input should be the name of the person.",
                return_direct=True
            ),
            Tool(
                name="Positive AI asistent",
                func=whoimi,
                description="Useful for when you need to answer questions about Positive AI asistent. Input should be Positive AI asistent "
            ),
        ]
    if 'agent_chain' not in st.session_state:
        st.session_state.agent_chain = initialize_agent(tools=st.session_state.tools,
                                                        llm=st.session_state.chat,
                                                        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
                                                        messages=st.session_state.chat_prompt,
                                                        verbose=False,
                                                        memory=st.session_state.memory,
                                                        handle_parsing_errors=True,
                                                        max_iterations=4
                                                        )
    if upit := st.chat_input("Postavite pitanje"):
        formatted_prompt = st.session_state.chat_prompt.format_prompt(
            text=upit).to_messages()
        # prompt[0] je system message, prompt[1] je tekuce pitanje
        pitanje = formatted_prompt[0].content+formatted_prompt[1].content

        with placeholder.container():

            odgovor = st.session_state.agent_chain.run(pitanje)
            time.sleep(1)
            st.session_state.stream_handler.clear_text()
            st.session_state.past.append(f"{name}: {upit}")
            st.session_state.generated.append(f"AI Asistent: {odgovor}")
            # Calculate the length of the list

            num_messages = len(st.session_state['generated'])
            # Loop through the range in reverse order
            for i in range(num_messages - 1, -1, -1):
                # Get the index for the reversed order
                reversed_index = num_messages - i - 1
                # Display the messages in the reversed order
                st.info(st.session_state["past"]
                        [reversed_index], icon="🤔")

                st.success(st.session_state["generated"]
                           [reversed_index], icon="👩‍🎓")
                # Append the messages to the download_str in the reversed order
                download_str.append(
                    st.session_state["past"][reversed_index])
                download_str.append(
                    st.session_state["generated"][reversed_index])
            download_str = '\n'.join(download_str)

        with st.sidebar:
            st.download_button('Download', download_str)

elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')
elif st.session_state["authentication_status"] is None:
    st.warning('Please enter your username and password')


