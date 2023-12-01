import streamlit as st
from langchain.agents import create_csv_agent
from langchain import OpenAI
from dotenv import load_dotenv
import os
import tempfile


def main():
    load_dotenv()

    st.set_page_config(page_title="Ask me something 🧞‍♂️")
    st.header("Ask me something 🧞‍♂️")

    user_csv = st.file_uploader("Upload a file", type=("csv"))

    if user_csv is not None:
        user_question = st.text_input("Ask me a question about the file")

        llm = OpenAI(temperature=0)
        
        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        
        # Write the contents of the uploaded file into the temporary file
        temp_file.write(user_csv.read())

        # Close the temporary file to ensure data is saved
        temp_file.close()

        agent = create_csv_agent(llm=llm, path=temp_file.name, verbose=True)

        if user_question is not None and user_question != "":
            output = agent.run(input=user_question)
            st.write(output)
            
        # Delete the temporary file after usage
        os.remove(temp_file.name)


if __name__ == '__main__':
    main()
