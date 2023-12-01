# pip install streamlit streamlit-chat langchain python-dotenv
import pandas as pd
import sqlite3
import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv
import os

#table_name = 'Employees'
#column_names = 'EEID,Name,Title,Department,Business_Unit,Gender,Ethnicity,Age,Hire_Date,Annual_Salary,Bonus,Country,City,Exit_Date'


from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

def init():
    # Load the OpenAI API key from the environment variable
    load_dotenv()
    
    # test that the API key exists
    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        print("OPENAI_API_KEY is not set")
        exit(1)
    else:
        print("OPENAI_API_KEY is set")

    # setup streamlit page
    st.set_page_config(
        page_title="Your own AskDB",
        page_icon="🤖"
    )

def run_chatbot(table, columns):
    template = """You are an application which converts human input text to SQL queries. 
    If you don't understand a user input you return 'Invalid query'

    Below are the details of the table you will be working on.
    
    Table Name: {table_name}
    Column Names: {column_names}

    {chat_history}
    Human: {human_input}
    Chatbot:"""

    # Update the input variable names to match the template
    prompt = PromptTemplate(input_variables=["table_name", "column_names", "chat_history", "human_input"], template=template)
    memory = ConversationBufferMemory(memory_key="chat_history", input_key="human_input")

    llm = OpenAI()
    llm_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=True,
        memory=memory,
    )

    while True:
        # Get input from the console/terminal
        human_input = input("You: ")

        # Check if the human input contains any variation of "exit" or "stop"
        if any(keyword in human_input.lower() for keyword in ["exit", "stop"]):
            print("Chatbot: Goodbye!")
            break

        # Predict the response using the provided human input
        response = llm_chain.predict(table_name=table, column_names=columns, chat_history="", human_input=human_input)
        print("Chatbot:", response)
        print()
        display_data_from_table(response)
        
def display_data_from_table(query):
    # Connect to the SQLite database
    conn = sqlite3.connect('demo1.db')

    # Create a cursor to interact with the database
    cursor = conn.cursor()

    # Execute query 
    cursor.execute(query)

    # Fetch all rows and print them
    rows = cursor.fetchall()
    for row in rows:
        print(row)

    # Close the cursor and the connection
    cursor.close()
    conn.close()

def main():
    table_name = 'Employees'
    column_names = 'EEID,Name,Title,Department,Business_Unit,Gender,Ethnicity,Age,Hire_Date,Annual_Salary,Bonus,Country,City,Exit_Date'

    init()
    run_chatbot(table_name, column_names)

if __name__ == "__main__":
    main()
   
