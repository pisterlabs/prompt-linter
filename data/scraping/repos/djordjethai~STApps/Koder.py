# This code is used to create a code based on LangChain Library using streamlit for web interface

from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import pinecone
import streamlit as st
import os
from mojafunkcja import st_style, positive_login, init_cond_llm

from langchain.callbacks.tracers.run_collector import RunCollectorCallbackHandler
from langchain.memory import StreamlitChatMessageHistory, ConversationBufferMemory
from langchain.schema.runnable import RunnableConfig
from langsmith import Client
from streamlit_feedback import streamlit_feedback
from langchain.callbacks.tracers.langchain import wait_for_all_tracers
from vanilla_chain import get_llm_chain
client = Client()

# these are the environment variables that need to be set for LangSmith to work
os.environ["LANGCHAIN_PROJECT"] = "Koder"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.langchain.plus"
os.environ.get("LANGCHAIN_API_KEY")


st.set_page_config(
    page_title="Koder",
    page_icon="👉",
    layout="wide"
)
st_style()



def main():
    # Set text field
    text_field = "text"

    # Insert path to PythonGPT3Tutorial

    # Read API keys from env
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

    # Retrieving API keys from env
    PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
    PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')

    # Initialize OpenAIEmbeddings and Pinecone
    embeddings = OpenAIEmbeddings()
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)

    # Create Pinecone index
    index = pinecone.Index("embedings1")
    name_space = "koder"
    vectorstore = Pinecone(index, embeddings,
                           text_field, name_space)

    # Get user input
    st.subheader("Koristeći LangChain i Streamlit 🔢")

    # Initialize ChatOpenAI and RetrievalQA

    st.session_state["izlaz"] = ""
    model, temp = init_cond_llm()
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY,
                     model_name=model, temperature=temp)
    qa = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever(), verbose=False)

    # Save the user input in the session state
    placeholder = st.empty()
    st.session_state["task"] = ""

    # Create a form with a text input and a submit button
    with placeholder.form(key="my_form", clear_on_submit=True):
        query = (
            "Using langchain and streamlite, "
            + st.text_area(
                label="detaljno opišite šta želite da uradim (kod, objašnjenje ili sl): ",
                key="1",
                value=st.session_state["task"],
            )
            + "."
        )
        submit_button = st.form_submit_button(label="Submit")

        # If the submit button is clicked, clear the session state and run the query
        if submit_button:
            st.session_state["task"] = ""
            with st.spinner("Sačekajte trenutak..."):
                st.session_state["izlaz"] = qa.run(query)
                st.write(st.session_state["izlaz"])

    if "izlaz" in st.session_state:
        st.download_button("Download as .txt",
                           st.session_state["izlaz"], file_name="koder.txt")


name, authentication_status, username = positive_login(main, "12.09.23. - Nemanja")


