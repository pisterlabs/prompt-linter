# Import libraries
import streamlit as st
import pandas as pd
import os
from langchain import OpenAI
from langchain.agents import create_csv_agent

# Streamlit page configurations and title
st.set_page_config(
    page_title="StudyGPT",
    page_icon=":mortar_board:",
    initial_sidebar_state = "collapsed"
)
st.title("📈 CSV Q&A")
st.caption("✨ Your personal CSV data assistant - upload and start asking questions!")

# Load API Key
api_key = st.secrets["OPENAI_API_KEY"]

# ------ Initialize Session State ------
if 'csv_response' not in st.session_state:
    st.session_state.csv_response = ''
    
if 'placeholder_initialized' not in st.session_state:
    st.session_state.placeholder_initialized = False

# ------ Helper functions ------
def save_uploaded_file(uploadedfile):
    with open(os.path.join("data", uploadedfile.name), "wb") as f:
        f.write(uploadedfile.getbuffer())
        
def clear_directory(directory):
    for file in os.listdir(directory):
        os.remove(os.path.join(directory, file))

# ------ Load and index document ------
uploaded_file = st.file_uploader(" ", accept_multiple_files=False,
                                  label_visibility='collapsed', type=['csv'])

# ------ Create agent and chat ------
if uploaded_file is not None:
    # Create data folder if it doesn't exist
    if not os.path.exists('./data'):
        os.mkdir('./data')
    else:
        clear_directory('./data')  # Clear the data directory

    # Save uploaded file to data folder
    save_uploaded_file(uploaded_file)

    # Load and index document
    uploaded_file_path = os.path.join('data', uploaded_file.name)
    
    # Create agent
    agent = create_csv_agent(OpenAI(temperature=0), uploaded_file_path, verbose=True)

    # Display uploaded CSV file as DataFrame
    df = pd.read_csv(uploaded_file)
    st.dataframe(df)

    if not st.session_state.placeholder_initialized:
        # Add a placeholder for the output
        output_placeholder = st.markdown("🤖 **AI:** I'm here to help you analyze this CSV! Ask me questions about the data, and I'll do my best to provide insights.\n\n")
        st.session_state.placeholder_initialized = True
    else:
        output_placeholder = st.empty()

    with st.form("chat_form", clear_on_submit=True):
        col1, col2 = st.columns([10, 1])
        user_prompt = col1.text_area(" ", max_chars=2000, key="prompt",
                                      placeholder="Type your question here...", label_visibility="collapsed")
        submitted = col2.form_submit_button("💬")

    if submitted and user_prompt:
        with st.spinner("💭 Waiting for response..."):
            st.session_state.csv_response = agent.run(user_prompt)
        response_md = f"🤓 **YOU:** {user_prompt}\n\n🤖 **AI:** {st.session_state.csv_response}\n\n"
        output_placeholder.markdown(response_md)  # Update the content of the placeholder