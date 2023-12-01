import streamlit as st
import pandas as pd
from langchain import OpenAI
from langchain.agents import create_csv_agent
import openai
import os


class TalkCSV:
    def __init__(self):
        self.df = None
        self.file_path = None
        self.agent = None
        self.question = ""
        self.answer = ""

    def run(self):
        st.title("TalkCSV: Talk to your CSV! 📄")
        st.write("Now you can ask questions about your CSV file and get answers! Just upload your CSV file and ask away!")

        self.sidebar()

        if self.df is not None:
            st.sidebar.write("File uploaded!")
            self.file_path = self.uploaded_file.name
            # st.write("Uploaded file path:", self.file_path)

        self.process_question()

        if self.answer:
            st.write("Answer:", self.answer)

    def sidebar(self):
        st.sidebar.title("OpenAI API Key 🔑")
        st.sidebar.write("You need an OpenAI API key to use this app. If you don't have one, you can get one [here](https://beta.openai.com/).")
        input_key = st.sidebar.text_input("Enter your OpenAI API Key here:", type="password")

        if st.sidebar.button("Submit") and input_key != "" and len(input_key) == 51:
            os.environ['OPENAI_API_KEY'] = input_key
            st.sidebar.write("Key submitted!")
        else:
            st.sidebar.write("Enter valid key!")

        self.uploaded_file = st.sidebar.file_uploader("Upload a CSV file 🗂️", type="csv")

        if self.uploaded_file is not None:
            self.df = pd.read_csv(self.uploaded_file)
            # check if the file is a valid CSV file
            try:
                self.df.head()
            except Exception as e:
                st.sidebar.write("Invalid CSV file!")
        else:
            st.sidebar.write("File not uploaded!")

    def process_question(self):
        try:
            openai_api_key = openai.api_key
            self.agent = create_csv_agent(OpenAI(temperature=0), self.file_path, verbose=False)

            self.question = st.text_input("Question:")
        except Exception as e:
            self.question = st.text_input("Question:")

        if st.button("Send"):
            try:
                self.answer = self.agent.run(self.question)
            except Exception as e:
                st.write(e)

        # if st.button("Clear"):
        #     self.question = ""
        #     self.answer = ""


if __name__ == "__main__":
    talk_csv = TalkCSV()
    talk_csv.run()
