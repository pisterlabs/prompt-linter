from langchain.agents import create_csv_agent
from langchain.llms import OpenAI
import os
from dotenv import load_dotenv
import streamlit as st
import tempfile

def main_ui2():
    st.header("Ask your CSV 📈")

    csv_file = st.file_uploader("Upload a CSV file", type="csv")
    if csv_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
            temp_file.write(csv_file.getvalue())
            temp_path = temp_file.name

        user_question = st.text_input("Ask a question about your CSV: ")

        if user_question is not None and user_question != "":
            with st.spinner(text="In progress..."):
                st.write("Processing... (Functionality not implemented)")

if __name__ == "__main__":
    # Sidebar for selecting UI
    selected_ui = st.sidebar.selectbox("Select UI:", ["UI 2"])

    if selected_ui == "UI 2":
        main_ui2()

def main():
    load_dotenv()
    # Load the OpenAI API key from the environment variable
    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        print("OPENAI_API_KEY is not set")
        exit(1)
    else:
        print("OPENAI_API_KEY is set")

    st.set_page_config(page_title="Ask your CSV")
    st.header("Ask your CSV 📈")
    
    csv_file = st.file_uploader("Upload a CSV file", type="csv")
    if csv_file is not None:
        # Create a temporary file and write the contents of the uploaded file into it
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
            temp_file.write(csv_file.getvalue())
            temp_path = temp_file.name

        agent = create_csv_agent(
            OpenAI(temperature=1), temp_path, verbose=True)

        user_question = st.text_input("Ask a question about your CSV: ")

        if user_question is not None and user_question != "":
            with st.spinner(text="In progress..."):
                st.write(agent.run(user_question))


if __name__ == "__main__":
    main()

