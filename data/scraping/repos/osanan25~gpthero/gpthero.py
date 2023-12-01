# Step-1 Write all the import statements from the Draft Code.
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

# Step-2 Write all the function definitions
def textRephraser(original_text):
    try:
        chat = ChatOpenAI(
            model="gpt-3.5-turbo-16k",
            temperature=0.7
    )
    system_template = """You are an assistant designed to rephrase the given text."""
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_template = """Please rephrase the following text: '{original_text}'."""
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chain = LLMChain(llm=chat, prompt=chat_prompt)
    result = chain.run(original_text=original_text)
    return result # returns string

    except Exception as e:
          st.error(f"An error occurred: {e}")
          return ""

def display_rephrased_text(rephrased_text):
    if rephrased_text != "":
        st.markdown(f"**Rephrased Text:** {rephrased_text}")

# Step-3 Get input from the user
st.title('GPTHero')
original_text = st.text_area("Enter the original text here")

# Step-4 Put a submit button with an appropriate title
if st.button('Rephrase Text'):
    # Step-5 Call functions only if all user inputs are taken and the button is clicked.
    if original_text:
        rephrased_text = textRephraser(original_text)
        display_rephrased_text(rephrased_text)
    else:
        st.warning('Please enter the original text to rephrase.')
else:
    rephrased_text = ""
