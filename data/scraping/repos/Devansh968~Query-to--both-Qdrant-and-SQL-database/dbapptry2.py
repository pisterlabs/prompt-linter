# pip install streamlit streamlit-chat langchain python-dotenv
import pandas as pd
import sqlite3
from streamlit_chat import message
from dotenv import load_dotenv
from langchain import SQLDatabase,SQLDatabaseChain,PromptTemplate
from langchain.memory import ConversationBufferMemory
from sqlalchemy import exc
from langchain import LLMChain
import streamlit as st
from langchain.chat_models import ChatOpenAI
import qdrant_client
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.vectorstores import Qdrant
from langchain.embeddings.openai import OpenAIEmbeddings
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
     # Initialize chat history as an empty list
    chat_history = []

    # Create the form outside the loop with a unique key
    form = st.form(key="chat_form")

    # Get the user input
    human_input = form.text_input("You:")

    # Check if the human input contains any variation of "exit" or "stop"
    if form.form_submit_button("Send"):
        if any(keyword in human_input.lower() for keyword in ["exit", "stop"]):
            st.write("Chatbot: Goodbye!")
        else:
            # Add the user input to the chat history
            chat_history.append(("Human", human_input))
            
            # Concatenate chat history for model input
            chat_history_str = "\n".join([f"{name}: {message}" for name, message in chat_history])

            # Predict the response using the provided chat history
            response = llm_chain.predict(table_name=table, column_names=columns, chat_history=chat_history_str, human_input=human_input)

            # Add the chatbot response to the chat history
            chat_history.append(("Chatbot", response))

            # Display the chat history in Streamlit's output area
            st.text_area("Chat History:", value="\n".join([f"{name}: {message}" for name, message in chat_history]))

            # Display the data from the table based on the chatbot's response
            display_data_from_table(response)
 

def display_data_from_table(query):
    if query.lower() == "invalid query":
        st.write("Chatbot: Invalid query. Please try again.")
    else:
        # Connect to the SQLite database
        conn = sqlite3.connect('demo1.db')

        # Create a cursor to interact with the database
        cursor = conn.cursor()

        try:
            # Execute the query
            cursor.execute(query)

            # Fetch all rows and print them
            rows = cursor.fetchall()
            if len(rows) > 0:
                st.write("Data from the Table:")
                for row in rows:
                    st.write(row)
            else:
                st.write("No results found.")
        except sqlite3.Error as e:
            st.write("An error occurred while executing the query:", str(e))
        finally:
            # Close the cursor and the connection
            cursor.close()
            conn.close()

   
            
#ask to pdf

def get_vector_store():
    
    client = qdrant_client.QdrantClient(
        os.getenv("QDRANT_HOST"),
        api_key=os.getenv("QDRANT_API_KEY")
    )
    
    embeddings = OpenAIEmbeddings()

    vector_store = Qdrant(
        client=client, 
        collection_name=os.getenv("QDRANT_COLLECTION_NAME"), 
        embeddings=embeddings,
    )
    
    return vector_store 



def main():
    table_name = 'Employees'
    column_names = 'EEID,Name,Title,Department,Business_Unit,Gender,Ethnicity,Age,Hire_Date,Annual_Salary,Bonus,Country,City,Exit_Date'
    load_dotenv()
    a =  st.sidebar.radio("Navigation",["Ask to your qdrant database","Query Database Like you Chat"])
    if a == "Ask to your qdrant database":
        #st.set_page_config(page_title="Ask")
        st.header("Ask to your qdrant database 💬")
    
    # creating vector store
        vector_store = get_vector_store()
    
    # create chain 
        qa = RetrievalQA.from_chain_type(
            llm=OpenAI(),
         chain_type="stuff",
            retriever=vector_store.as_retriever()
        )
    
    # show user input
        user_question = st.text_input("Ask a question about your PDF:")
        if user_question:
            st.write(f"Question: {user_question}")
            answer = qa.run(user_question)
            st.write(f"Answer: {answer}")
    else:
      
      init()
      run_chatbot(table_name, column_names)
if __name__ == "__main__":
    main()
   