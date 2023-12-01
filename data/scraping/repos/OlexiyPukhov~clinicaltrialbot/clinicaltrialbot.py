import streamlit as st
from streamlit_chat import message
import os
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
from langchain.tools import DuckDuckGoSearchRun
from langchain.utilities import PythonREPL
from langchain import OpenAI
from langchain.tools import BaseTool, StructuredTool, Tool, tool
from langchain.utilities import WikipediaAPIWrapper
from langchain.agents import AgentType, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.utilities import PythonREPL
from langchain.memory import ConversationBufferMemory

#def init():
#    load_dotenv()

password = os.environ.get('PASSWORD')

def check_password():
    if "password_entered" not in st.session_state:
        st.session_state.password_entered = False

    if not st.session_state.password_entered:
        password_form = st.sidebar.empty()

        with password_form.form(key='password_form'):
            user_password = st.text_input("Enter a password", type="password")
            submit_password = st.form_submit_button('Enter')

        if submit_password:
            if user_password == password:
                st.session_state.password_entered = True
                password_form.empty()  # clear the form
                st.experimental_rerun()    
            elif user_password != "":
                st.error("the password you entered is incorrect")
    else:
        return True

    return st.session_state.password_entered

def main():    

    #init()
    #st.set_page_config(
    #    page_title= "AI Researcher",
    #)
    
    custom_css = """
    <style>
        .reportview-container {
            background: #000000
        }
        .main footer, .main .block-container {
            color: #000000
        }
        .sidebar .sidebar-content {
            background: #333333
        }
    </style>
    """

    st.markdown(custom_css, unsafe_allow_html=True)


    openai_api_key = os.environ.get('OPENAI_API_KEY')
    

    st.image("ctbot.png", width=150)
    st.header("Recent Clinical Trials AI Chatbot")

    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
    
    def duck_wrapper(query):
        ddg_search = DuckDuckGoSearchRun() # create an instance of DuckDuckGoSearchRun
        return ddg_search._run(f"www.clinicaltrials.gov/ after:2021 {query}") # call the _run method with your query

    
    tools = [
        Tool(
            name = "Python",
            func=PythonREPL().run,
            description="useful for when you need to use python to answer a question. You should input python code"
        ),
         Tool(
        name='DuckDuckGo-Search',
        func= duck_wrapper,
        description="Useful to find information on anything"
    )
    ]
    
    zero_shot_agent = initialize_agent(
    agent=AgentType.OPENAI_FUNCTIONS,
    tools=tools, 
    llm=llm,
    verbose=True,
    max_iterations=4)
    




    
    if "messages" not in st.session_state:

        st.session_state.messages = [
                SystemMessage(content="System: Your job is to help the user answer various tax information by first asking questions relevant to the country of the city that the user inputs. Make sure you only respond with one question at a time. You do not have the capability to generate documents, but you can help the user fill out forms that they already have. Be sure to always list examples of answers to your question.",)
            ]
        message("Assistant: Hello! I'm your AI Assistant that can look up info on clinical trials > 2021 and summarize the information. How can I help?", is_user=False)

    
    with st.sidebar.form(key="message_form", clear_on_submit=True):
        user_input = st.text_input("Type your message here...", value="", key="user_input")
        submit_button = st.form_submit_button("Send")
        
   # with st.sidebar:
    #    user_input = st.text_input("Your message: ", key = f"user_input", value="")
    
    if submit_button:

        st.session_state.messages.append(HumanMessage(content=user_input))
        with st.spinner("Thinking..."): 
            
            response = zero_shot_agent.run(user_input)
        st.session_state.messages.append(AIMessage(content=response))
        
    messages = st.session_state.get("messages", [])
    
    for i, msg in enumerate(messages[1:]):
        if i % 2 == 0:
            message(msg.content, is_user=True, key = str(i) + "_user")
        else:
            message(msg.content, is_user = False, key = str(i) + "_ai")
            
        
        

if __name__ == "__main__":
    if check_password():
        main()