from dotenv import load_dotenv
import os
import pandas as pd
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import AzureChatOpenAI
import openai
from langchain import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.llms import AzureOpenAI
import tiktoken
import sqlite3
import fitz  # PyMuPDF
from sqlalchemy import create_engine
import json
import sympy as sp
# Tabular data preprocessing


class TabularDataProcessor:
    def __init__(self, data):
        self.data = data

    def preprocess(self):
        self.data = self.data.fillna('Unknown')  # Fill missing values
        self.data = self.data.applymap(lambda s: s.lower() if type(
            s) == str else s)  # convert text to lowercase

    # def preprocess_math_from_excel(self, file_path):
    #     # Read the Excel file
    #     data = pd.read_excel(file_path)

    #     # Define a function to process a mathematical expression
    #     def process_expression(expr):

    def get_data_from_db(self, query):
        self.data = pd.read_sql_query(query, self.engine)
        return self.data

    # processor = TabularDataProcessor('postgresql://username:password@localhost/dbname')
    # data = processor.get_data_from_db('SELECT * FROM tablename')
    # data = processor.preprocess_json('json_column_name)

    def preprocess_json(self, json_column):
        self.data[json_column] = self.data[json_column].apply(
            json.loads)  # load json string into json
        # Normalize the json to flat table
        self.data = pd.json_normalize(self.data[json_column])
        return self.data

    def merge_rows(self):
        self.data = self.data.groupby('column1').agg(lambda x: ' '.join(set))

    def transform_to_sentences(self):
        # Create a list to store the sentences
        sentences = []

        # Iterate over each row in the DataFrame
        for index, row in self.data.iterrows():
            # Create a sentence for each row
            sentence = ','.join(
                [f'{col} is {val}' for col, val in row.items()])
            sentences.append(sentence)

        return sentences

    def get_num_sheets(self, excel_file):
        # Get the number of sheets in the excel file
        return len(excel_file.sheet_names)

    def get_sheet_names(self, excel_file):
        # Get the name of the sheets in the excel file
        return excel_file.sheet_names

    def process_all_sheets(self, excel_file):
        # Initialize text
        text = ""

        # Iterate over all sheets in the excel file
        for sheet in excel_file.sheet_names:
            df = pd.read_excel(excel_file, sheet_name=sheet)

            # Process the data in the sheet
            self.data = df
            self.preprocess()
            text += " .".join(self.transform_to_sentences()) + ". "

        return text


def translate(text, target_language='ko'):
    # Use the translation API
    # This function should return translated text
    translated_text = text  # replace this with the translation API
    return translated_text


def main():
    # Establish a connection to the database (will create it if it doesn't exist)
    conn = sqlite3.connect('chat_history.db')

    # Create a cursor object
    c = conn.cursor()

    # Create table
    c.execute('''CREATE TABLE IF NOT EXISTS chat_history
                 (question TEXT, answer TEXT)''')

    # Save (commit) the changes
    conn.commit()

    st.set_page_config(page_title="Megazone Cloud ChatBot")
    st.markdown("<h1 style='text-align: center; color: lightgreen;'>Megazone Cloud ChatBot 💬</h1>",
                unsafe_allow_html=True)

    # load environment variables
    load_dotenv()

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
    # OPENAI_DEPLOYMENT_NAME = os.getenv("OPENAI_DEPLOYMENT_NAME")
    # OPENAI_MODEL_NAME = st.selectbox(
    #   'Select GPT Model', ('GPT35Turbo', 'GPT48K', 'GPT432K'))  # added model selection
    OPENAI_MODEL_NAMES = os.getenv("OPENAI_MODEL_NAMES").split(',')
    OPENAI_DEPLOYMENT_NAMES = os.getenv("OPENAI_DEPLOYMENT_NAMES").split(',')
    OPENAI_EMBEDDING_DEPLOYMENT_NAME = os.getenv(
        "OPENAI_EMBEDDING_DEPLOYMENT_NAME")
    OPENAI_EMBEDDING_MODEL_NAME = os.getenv("OPENAI_EMBEDDING_MODEL_NAME")
    OPENAI_DEPLOYMENT_VERSION = os.getenv("OPENAI_DEPLOYMENT_VERSION")
    OPENAI_MODEL_NAME = st.selectbox('Select GPT Model', OPENAI_MODEL_NAMES)
    OPENAI_DEPLOYMENT_NAME = st.selectbox(
        'Select GPT Deployment name', OPENAI_DEPLOYMENT_NAMES)

    # init Azure OpenAI
    openai.api_type = "azure"
    openai.api_version = OPENAI_DEPLOYMENT_VERSION
    openai.api_base = OPENAI_API_BASE
    openai.api_key = OPENAI_API_KEY
    # init openai
    llm = AzureChatOpenAI(deployment_name=OPENAI_DEPLOYMENT_NAME,
                          model_name=OPENAI_MODEL_NAME,
                          openai_api_base=OPENAI_API_BASE,
                          openai_api_version=OPENAI_DEPLOYMENT_VERSION,
                          openai_api_key=OPENAI_API_KEY)

    embeddings = OpenAIEmbeddings(
        deployment=OPENAI_EMBEDDING_DEPLOYMENT_NAME, model=OPENAI_EMBEDDING_MODEL_NAME, chunk_size=1)

    # Select chat mode
    natural_chat_mode = st.checkbox('Switch to Natural Chat Mode')

    if natural_chat_mode:
        prompt_template = st.text_input("Custom Prompt 🎯:")
        user_input = st.text_input("Type your message here 🤖:")
        # Create a placeholder for the chat history
        chat_placeholder = st.empty()

        # Fetch all records from the database
        c.execute("SELECT * FROM chat_history")
        rows = c.fetchall()

        # Display the chat history
        chat_history = "<h2>Chat History:</h2>"
        for row in rows:
            st.markdown(f"<strong>User :</strong> {row[0]}<br><strong>ChatBot :</strong> {row[1]}<br><br>",
                        unsafe_allow_html=True)

        chat_placeholder.markdown(chat_history, unsafe_allow_html=True)

        if user_input:
            response = openai.ChatCompletion.create(
                engine=OPENAI_DEPLOYMENT_NAME,
                messages=[
                    {"role": "system",
                     # "content": "Assistant is a large language model trained by OpenAI."},
                     "content": prompt_template},
                    {"role": "user", "content": user_input}
                ]
            )
            st.markdown(
                f'### Answer: \n {response["choices"][0]["message"]["content"]}', unsafe_allow_html=True)
            if st.button('Translate to Korean'):
                translated_text = translate(result)
            # Insert the question and answer into the database
            c.execute("INSERT INTO chat_history VALUES (?,?)",
                      (user_input, response["choices"][0]["message"]["content"]))

            # Commit the insert
            conn.commit()
            # Update the chat history with the new message
            chat_history = f"<strong>User :</strong> {user_input}<br><strong>ChatBot :</strong> {response['choices'][0]['message']['content']}<br><br>" + chat_history
            chat_placeholder.markdown(chat_history, unsafe_allow_html=True)

    else:

        # upload file
        uploaded_file = st.file_uploader("Upload your file", type=[
            "pdf", "csv", "txt", "xlsx", "xls"])

        # extract the text
        if uploaded_file is not None:
            file_details = {"FileName": uploaded_file.name,
                            "FileType": uploaded_file.type, "FileSize": uploaded_file.size}
            st.write(file_details)

            if file_details["FileType"] == "application/pdf":
                with st.spinner('Reading the PDF...'):
                    doc = fitz.open(
                        stream=uploaded_file.read(), filetype='pdf')
                    text = ""
                    for page in doc:
                        text += page.get_text()
                    # pdf_reader = PdfReader(uploaded_file)
                    # text = ""
                    # for page in pdf_reader.pages:
                    #     text += page.extract_text()

            elif file_details["FileType"] == "text/plain":
                with st.spinner('Reading the TXT file...'):
                    text = uploaded_file.read().decode("utf-8")

            elif file_details["FileType"] in ["application/vnd.ms-excel", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]:
                with st.spinner('Reading the Excel file...'):
                    excel_file = pd.ExcelFile(uploaded_file)

                    # text = process_all_sheets(excel_file)
                    # Create an instance of TabularDataProcessor
                    processor = TabularDataProcessor(None)
                    # df = pd.read_excel(uploaded_file)

                    # Get the number of sheets and their names
                    num_sheets = processor.get_num_sheets(excel_file)
                    sheet_names = processor.get_sheet_names(excel_file)

                    # sheet_names = excel_file.sheet_names

                    st.write(f"Number of sheets: {num_sheets}")
                    st.write(f"Sheet Names: {sheet_names}")

                    text = processor.process_all_sheets(excel_file)

                    # for sheet in sheet_names:
                    #     df = pd.read_excel(excel_file, sheet_name=sheet)
                    #     processor.data = df  # Set the data for the processor
                    #     processor.preprocess()
                    #     # text = " ".join(map(str, df.values))
                    #     text = ". ".join(
                    #         processor.transform_to_sentences()) + ""

            elif file_details["FileType"] == "text/csv":
                with st.spinner('Reading the CSV file...'):
                    df = pd.read_csv(uploaded_file)
                    processor = TabularDataProcessor(df)
                    processor.preprocess()
                    # text = " ".join(map(str, df.values))
                    text = ". ".join(processor.transform_to_sentences())

            else:
                st.error("File type not supported.")

            # split into chunks
            text_splitter = CharacterTextSplitter(
                separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
            chunks = text_splitter.split_text(text)

            # load the faiss vector store we saved into memory
            with st.spinner('Creating knowledge base...'):
                vectorStore = FAISS.from_texts(chunks, embeddings)

            # use the faiss vector store we saved to search the local document
            retriever = vectorStore.as_retriever(
                search_type="similarity", search_kwargs={"k": 2})

            # use the vector store as a retriever
            qa = RetrievalQA.from_chain_type(
                llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=False)

            # show user input
            prompt_template = st.text_input("Custom Prompt 🎯:")
            user_question = st.text_input("Ask a question 🤖:")

            chat_placeholder = st.empty()

            # Fetch all records from the database
            c.execute("SELECT * FROM chat_history")
            rows = c.fetchall()

            # Display the chat history
            chat_history = "<h2>Chat History:</h2>"
            for row in rows:
                st.markdown(f"<strong>User :</strong> {row[0]}<br><strong>Chat Bot :</strong> {row[1]}<br><br>",
                            unsafe_allow_html=True)

            chat_placeholder.markdown(chat_history, unsafe_allow_html=True)

            if user_question:
                result = qa({"query": user_question})
                # Display the result in a more noticeable way
                st.markdown(
                    f'### Answer: \n {result["result"]}', unsafe_allow_html=True)

                # Insert the question and answer into the database
                c.execute("INSERT INTO chat_history VALUES (?,?)",
                          (user_question, result["result"]))

                # Commit the insert
                conn.commit()
                chat_history = f"<strong>User :</strong> {user_question}<br><strong>ChatBot :</strong> {result['result']}<br><br>" + chat_history
                chat_placeholder.markdown(chat_history, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
