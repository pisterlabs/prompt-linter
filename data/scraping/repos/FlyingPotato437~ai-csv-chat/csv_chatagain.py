from langchain.memory import ConversationBufferMemory
from langchain.agents import create_csv_agent
from langchain.llms import OpenAI as LangchainOpenAI
from langchain.prompts import PromptTemplate
from streamlit_chat import message
import streamlit as st
import tempfile
import pandas as pd
from dotenv import load_dotenv

def main():
    load_dotenv()
    # Set the page layout
    st.set_page_config(layout='wide', page_title="CSVChat powered by OpenAI and created by Srikanth Samy 🤖")
    
    # Title of the application
    st.title("CSVChat powered by OpenAI 🤖")
    st.subheader("Created by Srikanth Samy")

    # File uploader for CSV files
    input_csv = st.file_uploader("Upload your CSV file", type=['csv'])

    # Initialize session state
    if 'past' not in st.session_state:
        st.session_state['past'] = []
        
    if 'generated' not in st.session_state:
        st.session_state['generated'] = []

    # If a CSV file is uploaded
    if input_csv is not None:
        # Create a temporary file and write the contents of the uploaded file into it
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
            temp_file.write(input_csv.getvalue())
            temp_path = temp_file.name

        # First column: display CSV data
        st.info("CSV Uploaded Successfully")
        data = pd.read_csv(input_csv)
        st.dataframe(data, use_container_width=True)

        # Initialize memory
        memory = ConversationBufferMemory(memory_key="chat_history")
        
        # Create LLM and agent
        llm = LangchainOpenAI(temperature=1)
        agent = create_csv_agent(llm, temp_path, verbose=True)

        # Container for the chat history
        response_container = st.container()

        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Query:", placeholder="Talk to your CSV data here (:", key='input')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            output = agent.run(user_input)
            st.session_state['past'].append(user_input)
            st.write("Debug Output:", output)  # Print the output for debugging
            # You will need to replace the next line with the correct way to access the answer from the output
            st.session_state['generated'].append(output)  # Temporarily appending the entire output for debugging


        if st.session_state['generated']:
            with response_container:
                for i in range(len(st.session_state['generated'])):
                    message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                    message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")

if __name__ == "__main__":
    main()
