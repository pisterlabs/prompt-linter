import streamlit as st
from langchain.agents import ConversationalChatAgent, AgentExecutor
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationBufferWindowMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.agents import tool
from typing import List
from langchain.tools import DuckDuckGoSearchRun
import os
import json
import openai

# Setting the configuration for the Streamlit page
st.set_page_config(
    page_title="Supplements Assistant",
    page_icon="💊",
)

# Title for the Streamlit page
st.title("💊 Supplements Assistant")

# Setting the OpenAI API key from environment variables
openai.api_key = os.environ["OPENAI_API_KEY"]

# Header for the user input section
st.header('_Fill in initial_ :blue[_info_]', divider='gray')

# Creating columns for user input
col1, col2, col3 = st.columns(3, gap="large")

# User input for gender
with col1:
  gender = st.radio("Select your gender: *", ["Male", "Female"])

# User input for age
with col2:
  age = st.number_input("Enter your age: *",
                        min_value=0,
                        max_value=120,
                        step=1)

# User input for pregnancy status, only if gender is Female
with col3:
  if gender == "Female":
    pregnant = st.radio("Are you pregnant: *", ["Yes", "No"])
  else:
    pregnant = 'No'

# User input for medical conditions
medical_cond = st.text_input("Outline your medical conditions: *")

# User input for describing a problem
problem = st.text_input("Describe your problem: *")

# Initialize an empty list to store additional information
additional_info = []

# Add pregnancy status to additional_info if applicable
if gender == "Female" and pregnant == "Yes":
  additional_info.append("I'm pregnant.")

# Add problem description to additional_info if provided
if problem:
  additional_info.append(f"I have this problem: {problem}.")

# Add medical conditions to additional_info if provided
if medical_cond:
  additional_info.append(f"Medical conditions: {medical_cond}.")

# Combining the user inputs to form an initial prompt
suggested_prompt = f"I'm {age} years old, {gender}. {' '.join(additional_info)} Can you recommend me what supplement should get?"

# Initialize a flag to check if the form has been submitted
form_submitted = False
# Handle the 'Submit' button action
if st.button("Submit"):
  # Validate if all the required inputs are provided
  if age and gender and pregnant and medical_cond and problem:
    st.success("Suggested prompt generated!")
    form_submitted = True  # Set the flag to True when the form is submitted
  else:
    st.warning("Please fill in all the required fields.")

if form_submitted:
  st.chat_message("assistant").write(
      f"_**Suggested prompt to ask**_ :green[_**Assistant**_]: {suggested_prompt}"
  )

# Header for the chatbot section
st.header('_Chat with_ :blue[_assistant_]', divider='gray')

# Initialize the chat message history
msgs = StreamlitChatMessageHistory()

# Initialize the conversation memory buffer
memory = ConversationBufferWindowMemory(k =1,
                                  chat_memory=msgs,
                                  return_messages=True,
                                  memory_key="chat_history",
                                  output_key="output")

# Reset chat history if there are no messages or the reset button is pressed
if len(msgs.messages) == 0 or st.sidebar.button("Reset chat history"):
  msgs.clear()
  msgs.add_ai_message(
      r"Hello, I'm your Supplements Assistant 👋"
  )
  msgs.add_ai_message(
      r"Would you like to know what supplement is best for the problem that you have?"
  )
  st.session_state.steps = {}

# Set avatars for user and AI
avatars = {"human": "user", "ai": "assistant"}

# Display previous chat messages
for idx, msg in enumerate(msgs.messages):
  with st.chat_message(avatars[msg.type]):
    # Render intermediate steps if any were saved
    for step in st.session_state.steps.get(str(idx), []):
      if step[0].tool == "_Exception":
        continue
      with st.status(f"**{step[0].tool}**: {step[0].tool_input}",
                     state="complete"):
        st.write(step[0].log)
        st.write(step[1])
    st.write(msg.content)

# Input field for the user to ask the chatbot a question
if prompt := st.chat_input(
    placeholder=
    "Based on the information I provided, what supplement would you recommend?"
):
  st.chat_message("user").write(prompt)
  
  sys_message = "You are a helpful assistant that answers ONLY questions about supplements, nutritions, health and sport. Please provide a detailed, comprehensive, and well-structured response to ensure clarity and thorough understanding. We value depth and thoroughness over brevity.  If user asks unrealated question, please mention that you were built to answer question about supplements, nutritions, health and sport. /n"
  
  # Initialize the chat model
  llm = ChatOpenAI(model_name="gpt-3.5-turbo",
                   openai_api_key=openai.api_key,
                   streaming=True)

  # List of tools the agent has access to
  tools = [DuckDuckGoSearchRun(name="Browser Search", description="useful ONLY when you need to answer questions about Reviews of a supplement")]
  #tools = [search, get_contents, find_similar]

  # Initialize the chat agent
  chat_agent = ConversationalChatAgent.from_llm_and_tools(llm=llm, tools=tools, system_message=sys_message)

  # Initialize the executor for the agent
  executor = AgentExecutor.from_agent_and_tools(agent=chat_agent,
                                                tools=tools,
                                                memory=memory,
                                                return_intermediate_steps=True,
                                                handle_parsing_errors=True)

  # Handle the agent's response in the chatbot UI
  with st.chat_message("assistant"):
    st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
    response = executor(prompt, callbacks=[st_cb])
    st.write(response["output"])
    st.session_state.steps[str(len(msgs.messages) -
                               1)] = response["intermediate_steps"]