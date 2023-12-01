

import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv
import os
import datetime
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
from streamlit_extras.colored_header import colored_header
from apscheduler.schedulers.background import BackgroundScheduler
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate
from langchain import LLMChain

st.set_page_config(
        page_title="Healing",
        page_icon="🤖"
        
    )


def robo_avatar_component():
    robo_html = "<div style='display: flex; flex-wrap: wrap; justify-content: center;'>"
    # replace with your own array of strings to seed the DiceBear Avatars API
    robo_avatar_seed = list(range(20, 25))

    for i in robo_avatar_seed:
        avatar_url = f"https://api.dicebear.com/5.x/bottts-neutral/svg?seed={i}"
        robo_html += f"<img src='{avatar_url}' style='width: 50px; height: 50px; margin: 30px;'>"
    robo_html += "</div>"

    robo_html = """
        <style>
        @media (max-width: 800px) {
            img {
                max-width: calc((100% - 60px) / 6);
                height: auto;
                margin: 0 10px 10px 0;
            }
        }
        </style>
        """ + robo_html

    st.markdown(robo_html, unsafe_allow_html=True)



# test user_input 
def send_message():
    now = datetime.datetime.now()
    print( now.hour, now.minute)
    return "who are you  "


def init():
    # Load the OpenAI API key from the environment variable
    load_dotenv()
    
    # test that the API key exists
    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        print("OPENAI_API_KEY is not set")
        exit(1)
    else:
        print("OPENAI_API_KEY is set")

    # setup streamlit page
    



from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate
from langchain import LLMChain

    
response_schemas = [
    ResponseSchema(name="options", description="a json array with {n_options} best option to solve the user question"),
    ResponseSchema(name="reason", description="the reason sentiment why it good behind user answer"),
    ResponseSchema(name="source", description="give lists the source used to answer the user's question, should be a website.")
]
template = """Answer the user question as best as possible, and you must speak in english.
{format_instructions}
The question: {question} """
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()
prompt = PromptTemplate(template=template, input_variables=["question"],partial_variables={"format_instructions": format_instructions})

def play_chat():
    init()
    chat = ChatOpenAI(temperature=0)
    solution_chain = LLMChain(llm=chat, prompt=prompt, output_key='result')
    # Initialize message history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content="You are a friendly chatbot decrease mental health in social media!!!. "),
            AIMessage(content="Give a note for user when their asking what you can do. You give functions that you can give a sentiment analysis for sentence and give your opinions to help user mental health")
        ]
    st.header("HEAl.AI 🤟")
    st.subheader("From Banh gio Team with love 🇻🇳")
    # Update the message history
    messages = st.session_state.get('messages', [])
    
    # Display the message history in reverse order (bottom-up)
    for i, msg in enumerate(messages):
        if isinstance(msg, SystemMessage):
            message(msg.content, is_user=False, key=str(i) + '_system')
        elif isinstance(msg, AIMessage):
            message(msg.content, is_user=False, key=str(i) + '_ai')
        elif isinstance(msg, HumanMessage):
            message(msg.content, is_user=True, key=str(i) + '_user')
    
    # build conservation bot 
    
    # Create a placeholder to display the response
    response_placeholder = st.empty()

    # Handle user input
    user_input = st.text_input("Your message: ")
    if st.button("Send"):
        user_input = user_input.strip()
        
        if user_input:
            with st.spinner("Thinking..."):
                # Use the sentiment mode
                #response = solution_chain({"question": user_input, "n_options": "3"})
                #sentiment = response["result"]                
                #sentiment = response['result'].strip().replace('```json', '').strip()
                #sentiment = sentiment.strip().replace('``', '').strip()
                # Generate the chatbot response
                response = chat(st.session_state.messages)
            st.session_state.messages.append(HumanMessage(content=user_input))
            st.session_state.messages.append(AIMessage(content=response.content))
            #st.session_state.messages.append(AIMessage(content=sentiment))
            # Update the response_placeholder with the new response
            #response_placeholder.text(sentiment)
            # Clear user_input after sending the message
            st.session_state.user_input = ""
        else:
            # If the user input is empty, keep the previous input in the text input field
            st.session_state.user_input = user_input
    robo_avatar_component()
    hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True) 