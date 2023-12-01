import time
import streamlit as st
from openai import OpenAI
import json
import os

# Initialize the OpenAI client
client = OpenAI()

# Retrieve an existing assistant
assistant = client.beta.assistants.retrieve(
    assistant_id="asst_g1yHeEvq8wPwdd9L7eMVrD6g"
)

if 'thread_id' not in st.session_state:
    # If not, create a new thread and store its ID in the session state
    thread = client.beta.threads.create()
    st.session_state['thread_id'] = thread.id

# Function to manage chat interaction
def chat_with_assistant(user_message):
    # Add a user message to the thread
    client.beta.threads.messages.create(
        thread_id=st.session_state['thread_id'],
        role="user",
        content=user_message
    )

    # Run the assistant on the thread
    run = client.beta.threads.runs.create(
        thread_id=st.session_state['thread_id'],
        assistant_id=assistant.id
    )

    # Check the run status and wait for completion
    while run.status not in ['completed', 'failed']:
        time.sleep(1)  # Sleep to prevent rapid polling
        run = client.beta.threads.runs.retrieve(
            thread_id=st.session_state['thread_id'],
            run_id=run.id
        )

    # Retrieve the messages from the thread after run completion
    if run.status == 'completed':
        messages = client.beta.threads.messages.list(
            thread_id=st.session_state['thread_id']
        )
        return [msg.content for msg in messages.data if msg.role == 'assistant']
    else:
        return ["Sorry, I couldn't complete the request."]

# Streamlit app layout enhancements
st.set_page_config(page_title="BitRoute Optimizer", layout="wide")

# Custom styles for branding and better UI
st.markdown("""
    <style>
    .app-header { visibility: hidden; }
    .stTextInput > label { font-size: 16px; font-weight: bold; }
    .stTextArea > label { font-size: 16px; font-weight: bold; }
    .chatbox { background-color: #fafafa; border-radius: 10px; padding: 10px; }
    .sidebar .sidebar-content { background-color: #1e1e1e; }
    </style>
    """, unsafe_allow_html=True)

# Sidebar with application information
st.sidebar.title("BitRoute Optimizer")
st.sidebar.markdown("⚡ Enhancing Bitcoin Transactions ⚡")
st.sidebar.markdown("""
- Efficient Pathfinding 🛣️
- Enhanced Privacy 🔒
- Cost-Effective 💰
""")

# Main chat interface
st.title("🤖 BitRoute Support Assistant")
st.markdown("Welcome to the BitRoute AI support chat. Ask me anything about optimizing your Bitcoin transactions!")

# User input
user_input = st.text_input("Enter your query:", max_chars=500)

# Chat display area
chat_history_container = st.empty()  # Placeholder to display chat history

# If user input is given, process the conversation
if user_input:
    # Get the assistant's response
    try:
        assistant_responses = chat_with_assistant(user_input)
        # Display the chat history with the response
        chat_history = chat_history_container.text_area(
            "Chat History",
            value=str(assistant_responses[0][0].text.value),
            height=300,
            key='chatbox',
            help="This is the history of your conversation with the AI assistant."
        )
        chat_history_container.markdown(chat_history, unsafe_allow_html=True)
    except TypeError as e:
        st.error(f"An error occurred: {e}")