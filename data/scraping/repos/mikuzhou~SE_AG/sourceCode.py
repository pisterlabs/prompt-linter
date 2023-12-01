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

st.title("useful at vandy")

# Initialize user inputs
article = ""
word_count = ""
expanded_condensed_article = ""

# Define articleModifier function
def articleModifier(article, word_count):
    chat = ChatOpenAI(
        model="gpt-3.5-turbo-16k",
        temperature=0
    )
    system_template = """You are an assistant tasked with expanding or condensing an article to a specific word count. The article is: {article}"""
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_template = """Please modify the given article to have a word count of {word_count}. The article is: {article}"""
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chain = LLMChain(llm=chat, prompt=chat_prompt)
    result = chain.run(article=article, word_count=word_count)
    return result # returns string   

# Get the article from the user
article = st.text_area("Enter the article")

# Get the desired word count from the user
word_count = st.text_input("Enter the desired word count")

# Check if all user inputs are taken and the button is pressed
if st.button("Expand/Condense"):
    if article and word_count:
        expanded_condensed_article = articleModifier(article, word_count)

# Display the expanded or condensed article to the user
st.markdown(expanded_condensed_article)
