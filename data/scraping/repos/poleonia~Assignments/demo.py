import streamlit as st
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (ChatPromptTemplate,
                                    HumanMessagePromptTemplate,
                                    SystemMessagePromptTemplate)
from langchain.document_loaders import *
from langchain.chains.summarize import load_summarize_chain
import tempfile
from langchain.docstore.document import Document

st.title("Lining.yu_CS8395_HW0_v1")

# Initialize user inputs
email_text = ""
keywords = ""

# Get the email text from the user
email_text = st.text_area("Enter email text")

# Get the keywords describing attitude against the text from the user
keywords = st.text_area("Enter keywords describing attitude against the text")

# Generate a reply to the email based on the input text and keywords
def emailReplyGenerator(email_text, keywords):
    chat = ChatOpenAI(
        model="gpt-3.5-turbo-16k",
        temperature=0.7
    )
    system_template = """You are an AI assistant designed to generate a reply to an email based on the input text and keywords."""
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_template = """Please generate a reply to the email using the provided text: '{email_text}' and keywords: {keywords}."""
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chain = LLMChain(llm=chat, prompt=chat_prompt)
    result = chain.run(email_text=email_text, keywords=keywords)
    return result # returns string   

# Check if all user inputs are taken and the button is pressed
if st.button("Generate Reply"):
    if email_text and keywords:
        reply = emailReplyGenerator(email_text, keywords)
        # Display the generated reply to the user
        st.markdown(reply)
    else:
        st.markdown("")