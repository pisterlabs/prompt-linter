import streamlit as st
import random
import time
from langchain.chains import TransformChain, LLMChain, SimpleSequentialChain, create_tagging_chain, create_tagging_chain_pydantic
from langchain.prompts import PromptTemplate, ChatPromptTemplate

from langchain.chat_models import ChatOpenAI
from enum import Enum
from pydantic import BaseModel, Field, conlist
from typing import Optional, Tuple

import plotly.express as px
import pandas as pd

import os
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

def radar_chart(motivation, education, career):  
    df = pd.DataFrame(dict(
    r=[motivation,
       education[0],
       education[1],
       education[2],
       career[0],
       career[1],
       career[2]
       ],
    theta=['Motivation', 'Highest Degree','Academic Major','College Ranking',
           'Job Level', 'Job Position', 'Company Ranking']))

    fig = px.line_polar(df, r='r', theta='theta',  line_close=True,
                    color_discrete_sequence=px.colors.sequential.Plasma_r,
                    template="plotly_dark", title="Candidate's Job Match", range_r=[0,10])
    st.sidebar.header('For Recruiter Only:')
    st.sidebar.write(fig)

class PersonalDetails(BaseModel):
    full_name: Optional[str] = Field(
        None,
        description="Is the full name of the user.",
    )
    
    school_background: Optional[conlist(int, min_items=3, max_items=3)] = Field(
        None,
        description="""Qualification level of education background. Range is 1 to 10, the bigger number the higher qualified.
        The first element indicates the level of degree, 10 means master degree or higher, 1 means high school.
        The second element indicates the major relevance, 10 means computer science and its releated major.
        The third element indicates the college ranking, 10 means the Top 50 college of world, 1 means community college.
        0 means indeterminated.
        """,
    )
    working_experience: Optional[conlist(int, min_items=3, max_items=3)] = Field(
        None,
        description="""Qualification status of career background.Range is 1 to 10, the bigger number the higher qualified.
        The first element indicates job level, 10 means senior manager or above, 1 means intern.
        The second element indicates position relevance, 10 means software development positions.
        The third element indicates the company Ranking, 10 means the Top 500 companies of world, 1 means small local company.
        0 means indeterminated.
        """,

    )
    interview_motivation: Optional[int] = Field(
        None,
        description="""The candidate's motivation level to join the interview.
        10 means very interested and enthusiastic about the interview and new role opening. 1 means not interested.
        """,
    )
    

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")

tagging_chain = create_tagging_chain_pydantic(PersonalDetails, llm)

def check_what_is_empty(user_peronal_details):
    ask_for = []
    # Check if fields are empty
    for field, value in user_peronal_details.dict().items():
        if value in [None, "", 0, "unknown"]:  # You can add other 'empty' conditions as per your requirements
            print(f"Field '{field}' is empty.")
            ask_for.append(f'{field}')
    return ask_for


## checking the response and adding it
def add_non_empty_details(current_details: PersonalDetails, new_details: PersonalDetails):
    non_empty_details = {k: v for k, v in new_details.dict().items() if v not in [None, "", 0, "unknown"]}
    updated_details = current_details.copy(update=non_empty_details)
    return updated_details

def ask_for_info(ask_for):

    # prompt template 1
    first_prompt = ChatPromptTemplate.from_template(
        """You are a job recruter who only ask questions.
        What you asking for are all and should only be in the list of "ask_for" list. 
        After you pickup a item in "ask for" list, you should extend it with 20 more words in your questions with more thoughts and guide.
        You should only ask one question at a time even if you don't get all according to the ask_for list. 
        Don't ask as a list!
        Wait for user's answers after each question. Don't make up answers.
        If the ask_for list is empty then thank them and ask how you can help them.
        Don't greet or say hi.
        ### ask_for list: {ask_for}

        """
    )

    # info_gathering_chain
    info_gathering_chain = LLMChain(llm=llm, prompt=first_prompt)
    ai_chat = info_gathering_chain.run(ask_for=ask_for)

    return ai_chat

def filter_response(text_input, user_details):
    #chain = create_tagging_chain_pydantic(PersonalDetails, llm)
    res = tagging_chain.run(text_input)
    #st.write("tag chain: " + str(res))
    # add filtered info to the
    user_details = add_non_empty_details(user_details,res)
    ask_for = check_what_is_empty(user_details)
    return user_details, ask_for

user_init_bio = PersonalDetails(full_name="",
                                school_background=None,
                                working_experience=None,
                                interview_motivation=0)
ask_init = ['full_name', 'school_background', 'working_experience', 'interview_motivation']

st.title("Yeyu's Interview Chatbot")
st.write("*for Senior Software Manager*")


# Initialize chat history

if "messages" not in st.session_state:
    question = ask_for_info(ask_init)
    st.session_state.messages = [{"role":"assistant", "content":question}]
if "bio_filled" not in st.session_state:
    st.session_state.bio_filled = user_init_bio
if "bio_missing" not in st.session_state:
    st.session_state.bio_missing = ask_init



# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if answer := st.chat_input("Please answer the question. "):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": answer})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(answer)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        st.session_state.bio_filled, st.session_state.bio_missing = filter_response(answer, st.session_state.bio_filled)
        #st.write(str(st.session_state.bio_filled))
        #st.write(str(st.session_state.bio_missing))
        if st.session_state.bio_missing != []:
            assistant_response = ask_for_info(st.session_state.bio_missing)
            
        else:
            assistant_response = """Thank you for participating in this interview. 
                                    We will notify you of the next steps once we have reached a conclusion.
                                 """
        

            final_details = st.session_state.bio_filled.dict()
            radar_chart(final_details['interview_motivation'], final_details['school_background'], final_details['working_experience'])
        # Simulate stream of response with milliseconds delay

        for chunk in assistant_response.split():
            full_response += chunk + " "
            time.sleep(0.05)
            # Add a blinking cursor to simulate typing
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
        
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    
    

