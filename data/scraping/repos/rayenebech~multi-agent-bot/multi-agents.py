from langchain.prompts.chat import HumanMessagePromptTemplate 
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage
from langchain import HuggingFaceHub
from dotenv import load_dotenv
import streamlit as st 
import time
import os

from helpers import build_prompt_from_database
from agent import EmployeeAgent

load_dotenv()

# open_llm = HuggingFaceHub(
#             repo_id="google/flan-t5-base",
#             model_kwargs={"temperature":0},
#         )

llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0,
    model_name='gpt-3.5-turbo'
)

############################################ Streamlit Config   #############################################################
st.title(':blue[Uptimes Lab]') 
st.header(' Incident Response Simulation')
username = "Rayene"


#################################################################################################################################

# Build the prompt from the database
system_prompt = build_prompt_from_database("database.json", username)
sys_msg = SystemMessage(content= system_prompt)
# print(sys_msg.content)

# initialize agent
agent = EmployeeAgent(sys_msg, llm = llm)

chat_turn_limit = 10
timeout_seconds = 60  # timeout after 60 seconds of no user input
for n in range(chat_turn_limit):
    user_text = st.text_input("your message: ", key=f"message_{n}")
    # Check if user has inputted a message
    while not user_text:
        pass
     # Display user's message
    st.write("")

    print(f"******************")
    print("user_text: ", user_text)
    print(f"******************")
    human_message = "{username}: {user_text}"
    human_template = HumanMessagePromptTemplate.from_template(template=human_message)
    user_msg = human_template.format_messages(username=username,user_text=user_text)[0]
    print(f"******************")
    print(f"{username}: {user_msg.content}")

    ai_msg = agent.step(user_msg)
    print(f"{ai_msg.content}")
    
    ## DISPLAY SECTION

    st.subheader("AI")
    st.write(ai_msg.content)


########################################################################################################