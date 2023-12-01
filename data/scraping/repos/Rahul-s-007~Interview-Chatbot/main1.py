import streamlit as st

st.set_page_config(page_icon=":computer:", layout = "wide")
st.write("<div style='text-align: center'><h1><em style='text-align: center; color:#00FFFF;'>Interview Practice</em></h1></div>", unsafe_allow_html=True)
#----------------------------------------------------------------#
from streamlit_chat import message
from streamlit import session_state

# st.session_state["messages"] = []
# st.session_state["messages"].append(message("Hello, I am your interviewer. What is your name?", "bot"))
if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "past" not in st.session_state:
    st.session_state["past"] = []
if "input" not in st.session_state:
    st.session_state["input"] = ""

#----------------------------------------------------------------#
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI, VectorDBQA
from langchain.document_loaders import DirectoryLoader
# import magic
import nltk
#----------------------------------------------------------------#
import openai

import os
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

openai.api_key = OPENAI_API_KEY # for open ai
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY # for lang chain

# @cache load model
#----------------------------------------------------------------#
def get_text():

    input_text = st.text_input("You: ", st.session_state["input"], key="input",
                            placeholder="Enter your response here...", 
                            label_visibility='hidden')
    return input_text

def chat():
    # input_text = st.text_input("You:", key="input_text")
    # if input_text:
    #     st.session_state["messages"].append(message(input_text, "user"))
    #     st.session_state["messages"].append(message("ans", "bot"))
    convo_col,temp = st.columns([1,10])

    user_input = get_text()

    if user_input:
        output = "ans"  
        st.session_state.past.append(user_input)
        st.session_state.generated.append(output)

    with temp:
        with st.expander("Conversation", expanded=True):
            if len(st.session_state["generated"]) !=0 and len(st.session_state["past"]) !=0:
                for i in range(0,len(st.session_state['generated'])):
                    st.success("Hope: "+st.session_state["generated"][i])  # icon="🤖"
                    st.info("User: "+st.session_state["past"][i])

#----------------------------------------------------------------#
def chat_software_developer():
    chat()

#----------------------------------------------------------------#
def chat_sales_representative():
    pass

#----------------------------------------------------------------#
def chat_marketing_manager():
    pass

#----------------------------------------------------------------#
def chat_data_scientist():
    pass

#----------------------------------------------------------------#
def chat_human_resources_manager():
    pass

#----------------------------------------------------------------#
def chat_project_manager():
    pass

#----------------------------------------------------------------#
def chat_financial_analyst():
    pass

#----------------------------------------------------------------#
def chat_customer_service_representative():
    pass

#----------------------------------------------------------------#
def chat_graphic_designer():
    pass

#----------------------------------------------------------------#
def chat_healthcare_administrator():
    pass

#----------------------------------------------------------------#
def chat_lawyer():
    pass

#----------------------------------------------------------------#
def chat_teacher():
    pass

#----------------------------------------------------------------#
def chat_web_developer():
    pass

#----------------------------------------------------------------#
job_roles = ["Software Developer",
             "Sales Representative",
             "Marketing Manager", 
             "Data Scientist", 
             "Human Resources Manager",
             "Project Manager",
             "Financial Analyst",
             "Customer Service Representative",
             "Graphic Designer",
             "Healthcare Administrator",
             "Lawyer",
             "Teacher",
             "Web Developer"]
#----------------------------------------------------------------#
def main():
    st.sidebar.write("Choose a role to be interviewed for:")
    options = st.sidebar.radio("Select Role",job_roles,label_visibility="collapsed")
    if options == "Software Developer":
        chat_software_developer()

    if options == "Sales Representative":
        chat_sales_representative()

    if options == "Marketing Manager":
        chat_marketing_manager()
        
    if options == "Data Scientist":
        chat_data_scientist()

    if options == "Human Resources Manager":
        chat_human_resources_manager()
 
    if options == "Project Manager":
        chat_project_manager()

    if options == "Financial Analyst":
        chat_financial_analyst()

    if options == "Customer Service Representative":
        chat_customer_service_representative()

    if options == "Graphic Designer":
        chat_graphic_designer()

    if options == "Healthcare Administrator":
        chat_healthcare_administrator()
        
    if options == "Lawyer":
        chat_lawyer()

    if options == "Teacher":
        chat_teacher()

    if options == "Web Developer":
        chat_web_developer()

    if st.sidebar.button("Clear Chat"):
        st.session_state["messages"] = []

#----------------------------------------------------------------#
if __name__ == "__main__":
    main()