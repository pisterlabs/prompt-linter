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

st.title("Deadlines")

# Initialize user inputs
assignment_description = ""
deadline = ""
start_date = ""

# Get the assignment description from the user
assignment_description = st.text_area("Assignment Description")

# Get the deadline for the assignment from the user
deadline = st.text_input("Deadline for the assignment")

# Get the start date for the assignment from the user
start_date = st.text_input("Start Date for the assignment")

# Initialize sub_assignments
sub_assignments = ""

# Generate sub-assignments and sub-deadlines based on the assignment description, deadline, and start date
def assignmentPlanner(assignment_description, deadline, start_date):
    chat = ChatOpenAI(
        model="gpt-3.5-turbo-16k",
        temperature=0
    )
    system_template = """You are an assistant tasked with generating sub-assignments and sub-deadlines based on the given assignment description, deadline, and start date."""
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_template = """Based on the assignment description: '{assignment_description}', deadline: '{deadline}', and start date: '{start_date}', please generate sub-assignments and their corresponding sub-deadlines."""
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chain = LLMChain(llm=chat, prompt=chat_prompt)
    result = chain.run(assignment_description=assignment_description, deadline=deadline, start_date=start_date)
    return result # returns string   

# Check if all user inputs are provided and the button is pressed
if assignment_description and deadline and start_date and st.button("Generate Sub-Assignments"):
    sub_assignments = assignmentPlanner(assignment_description, deadline, start_date)

# Display the sub-assignments and sub-deadlines to the user
st.markdown(sub_assignments)