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
    st.session_state["input"] = ''

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

scores_and_suggestions = []

next_question = ['Give any one of the questions the Interviewer asked or cross question based on the users previous answer.', 'wassup']

# @cache load model
#----------------------------------------------------------------#
def get_text():

    input_text = st.text_input("You: ", st.session_state["input"], key="input",
                            placeholder="Enter your response here...", 
                            label_visibility='hidden',on_change=submit)
    return input_text

#st.button("Next Question")
# as soon as a text is inputed show an end interview button

if 'something' not in st.session_state:
    st.session_state.something = ''

def submit():
    global input_text
    st.session_state.something = st.session_state.input
    st.session_state.input = ''
    # user_input = st.session_state.something
    
def chat():
    convo_col, temp, ques= st.columns([1, 10, 3])

    user_input = get_text()
    user_input = st.session_state.something
    

    output = "ans"  
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

    response = openai.Completion.create(
        engine="davinci",
        prompt=f"Q: {st.session_state.past[-1]}\nA: {st.session_state.generated[-1]}\nScore and suggestion:",
        temperature=0,
        max_tokens=100
    )

    if "scores_and_suggestions" not in st.session_state:
        st.session_state["scores_and_suggestions"] = []
        
    if 'next_question' not in st.session_state:
        st.session_state['next_question'] = []
        
    if 'next_question_index' not in st.session_state:
        st.session_state.next_question_index = 0
        
        
    score_and_suggestion = response.choices[0].text.strip()
    st.session_state.scores_and_suggestions.append(score_and_suggestion)


    st.session_state.stop_button_visible = True  # make Stop Interview button visible



    if not hasattr(st.session_state, 'stop_button_visible'):
        st.session_state.stop_button_visible = False  # hide Stop Interview button by default

    with temp:
        with st.expander("Conversation", expanded=True):
            if len(st.session_state["generated"]) != 0 and len(st.session_state["past"]) != 0:
                for i in range(0, len(st.session_state['generated'])):
                    st.success("AI: "+next_question[st.session_state.next_question_index])  # icon="🤖"
                    st.session_state.next_question_index += 1

                while True:
                    text = get_text()
                    if text != input_text:
                        input_text = text
                        print("User:", input_text)
                
        if st.session_state.stop_button_visible:  # show Stop Interview button if visible
            if st.button("Stop Interview"):
                with st.expander("Scores and Suggestions", expanded=True):
                    for i in range(len(st.session_state.past)):
                        st.write(st.session_state.scores_and_suggestions[i])

        with ques:
            if st.session_state.stop_button_visible:  # show Next Question button if visible
                if len(st.session_state["generated"]) != 0 and len(st.session_state["past"]) != 0:
                    next_question_index = len(st.session_state['next_question'])
                    if next_question_index < len(st.session_state['next_question']):
                        st.write(st.session_state.next_question[next_question_index])
                        st.session_state.next_question_index = next_question_index + 1  # increment the index
                        st.session_state.next_question.append("")  # add an empty string to the list to keep track of progress
                    # elif next_question_index == len(st.session_state['next_question']):
                    #     st.write("No more questions.")  # display a message if there are no more questions
                if st.button("Next Question"):
                    next_question_index = len(st.session_state['next_question'])
                    if next_question_index < len(st.session_state['next_question']):
                        st.write(st.session_state.next_question[next_question_index])
                        st.session_state.next_question_index = next_question_index + 1  # increment the index
                        st.session_state.next_question.append("")  # add an empty string to the list to keep track of progress
                    # elif next_question_index == len(st.session_state['next_question']):
                    #     st.write("No more questions.")  # display a message if there are no more questions



                    
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