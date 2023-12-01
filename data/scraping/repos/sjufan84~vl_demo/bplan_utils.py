""" Business Plan Chat Bot """
import os
import base64
from PIL import Image
import streamlit as st
import openai
from dotenv import load_dotenv
import pinecone
from langchain.chat_models import ChatOpenAI
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings

# Load the environment variables
load_dotenv()

# Load the OpenAI API key
openai.api_key = os.getenv("OPENAI_KEY2")
openai.organization = os.getenv("OPENAI_ORG2")

# Load the Pinecone API key
pinecone_key = os.getenv("PINECONE_KEY")
pinecone_env = os.getenv("PINECONE_ENV")

# Initialize pinecone
pinecone.init(api_key=pinecone_key, environment=pinecone_env)

# Initialize the embeddings
embed = OpenAIEmbeddings(openai_api_key=openai.api_key, openai_organization=openai.organization)

# Establish chat history and default model
if "messages" not in st.session_state:
    st.session_state.messages = []
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4-0613"

# Function to convert image to base64
def img_to_base64(img_path):
    """Convert an image to base64"""
    with open(img_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Use the function
IMG_PATH = "./resources/business_plan_bob.png"  # Replace with your image's path
#base64_string = img_to_base64(IMG_PATH)
business_bob = Image.open(IMG_PATH)

def get_context(query: str):
    """ Get the context from the business plan vector database."""
    index = Pinecone.from_existing_index("bplan", embedding=embed)
    retriever = index.as_retriever()

    llm = ChatOpenAI(temperature=0, model="gpt-4-1106-preview",
    openai_api_key=openai.api_key, openai_organization=openai.organization)

    #llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
    compressor = LLMChainExtractor.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)

    compressed_docs = compression_retriever.get_relevant_documents(f"{query}")
    return {"page_content": str([doc.page_content for doc in compressed_docs]), "metadata" : [doc.metadata for doc in compressed_docs]}

def get_new_prompt(query: str, model:str="openai"):
    """ Get a new prompt from the OpenAI API """
    if model == "openai":
        context = get_context(query)
    initial_message = {
        "role": "system",
        "content": f"""
        You are a Business Plan Bob, a master businessperson, entreprenuer, investor,
        and start-up advisor, advising a company called "First Rule"
        who is helping a start-up aiming to help artist's protect
        their "Melodic Voiceprint" as well as help them monetize
        it for downstream use cases.  A potential investor is browing
        their demo and wants to ask questions about their business plan.
        You have access to the business plan via a vector database that
        you can query with a search term.  The context returned from the 
        database is: {context}.  The query is {query}.  Your recent chat history
        is {st.session_state.messages[-2:] if len(st.session_state.messages) > 2 else None}.
        You are a representative of the company, so answer the investor's questions
        on their behalf.  Do not break character and help them relay their plan in a way
        that will help them get funding.
        """
    }

    return initial_message

def business_chat():
    """ Chat bot to answer questions about the business plan """
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        if message["role"] == "assistant":
            with st.chat_message(message["role"], avatar=business_bob):
                st.markdown(message["content"])
        elif message["role"] == "user":
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("What questions can I answer about First Rule's business plan?"):
        with st.spinner("Consulting the business plan..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)
            # Load the prophet image for the avatar
            # Display assistant response in chat message container
            with st.chat_message("assistant", avatar=business_bob):
                message_placeholder = st.empty()
                full_response = ""

            initial_message = [get_new_prompt(prompt)]

            for response in openai.ChatCompletion.create(
                model="gpt-4-1106-preview",
                messages=initial_message,
                stream=True,
                temperature=0.75,
                max_tokens=500,
                ):
                full_response += response.choices[0].delta.get("content", "")
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

def chat_state_sidebar():
    """ Sidebar for the chat state """
    # Create a radio button to select the page
    chat_state = st.sidebar.radio('**:rainbow[Questions about the Business Plan?  Turn on "Business Chat" and get answers from our AI Business Advisor.]**', ("on", "off"), index=1)
    if chat_state == "on":
        st.session_state.chat_state = "on"
        business_chat()
    else:
        st.session_state.chat_state = "off"
    # Create a "Clear Chat" button
    if st.sidebar.button("Clear Chat"):
        st.session_state.messages = []
