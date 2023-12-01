# -*- coding: utf-8 -*-
"""
@author: Sarick
"""
from dotenv import load_dotenv
load_dotenv()
import re
import os
import time
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine
from langchain.agents import create_sql_agent
from langchain.utilities import SQLDatabase
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType
from prompt import SQL_PREFIX, SQL_SUFFIX, PYTHON_PROMPT, FORMAT_INSTRUCTIONS
from python_utils import generate_code, format_primer, generate_primer

import langchain
from langchain.cache import SQLiteCache

langchain.llm_cache = SQLiteCache(
    database_path=".langchain.db"
)  # caches queries that are the same.


class SQLAgent:
    """
    SQLAgent class handles user queries, converts natural language into SQL queries,
    gets the answer, and passes the dataframe onto pythonrepl for code creation
    for visualization in Streamlit.
    """

    def __init__(self):
        """
        Initialize the SQLAgent with database connection and toolkit setup.
        """
        self.db_uri = "sqlite:///my_lite_store.db"
        self.db_instance = SQLDatabase.from_uri(self.db_uri)
        self.openai_api_key = os.getenv("OPEN_API_KEY")
        self.toolkit = SQLDatabaseToolkit(
            db=self.db_instance,
            llm=ChatOpenAI(temperature=0, model=st.session_state["selected_model"],
                           openai_api_key = self.openai_api_key),
        )
        
        self.agent_executor = create_sql_agent(
            llm=ChatOpenAI(temperature=0, model=st.session_state["selected_model"],
                           openai_api_key = self.openai_api_key),
            toolkit=self.toolkit,
            verbose=True,
            prefix=SQL_PREFIX,
            suffix=SQL_SUFFIX,
            format_instructions=FORMAT_INSTRUCTIONS,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            agent_executor_kwargs={"return_intermediate_steps": True},
            max_iterations=5,
        )

    def generate_query_response(self, message):
        """

        Parameters
        ----------
        message (str):
            The users query

        Returns
        -------
        split_string[0] (str)
            the natural language answer to the query
        split_string[1] (str)
            the sql portion of the query that needs to be further processed

        """
        response = self.agent_executor(message)
        output = response["output"]
        split_string = output.split("Here is the SQL query to obtain the results:")
        split_string[1] = self.extract_sql_code(split_string[1])
        return split_string[0], split_string[1]

    @staticmethod
    def extract_sql_code(text):
        """
        extract sql from the agent response

        Parameters
        ----------
        text (str): The text output of the agent

        Returns
        -------
        sql_match.group(1) (str) the sql only portion of the response

        """
        sql_match = re.search(r"```SQL\n(.+?)\n```", text, re.DOTALL)
        if sql_match:
            return sql_match.group(1)
        return "No SQL code found"

    def generate_dataframe(self, query):
        """
        use the agent generated sql query to get a dataframe for charting purposes


        Parameters
        ----------
        query (str):
            The extracted sql query from above

        Returns
        -------
        df  (Pandas DataFrame):
            The sql query output in a pandas dataframe of the agent generated query

        """
        engine = create_engine(self.db_uri)
        df = pd.read_sql_query(query, engine)
        return df
    
    
    def table_previews(self):
        """

        Returns
        -------
        A streamlit preview of the first 5 rows of each table.

        """
        engine = create_engine(self.db_uri)
        enrollments_preview = pd.read_sql_query(
            "select * from enrollments limit 5", engine
        )
        demographics_preview = pd.read_sql_query(
            "select * from county_demographics limit 5", engine
        )

        st.sidebar.write("Enrollments Data Preview")
        st.sidebar.dataframe(enrollments_preview)
        st.sidebar.write("County Demographics Data Preview")
        st.sidebar.dataframe(demographics_preview)

    def main(self):
        """
        Main function to handle user input and generate responses.
        """
        MODEL = st.sidebar.selectbox(
            label="Model", options=["gpt-3.5-turbo-16k-0613", "gpt-4"]
        )
        st.session_state["selected_model"] = MODEL
        user_input = st.text_input("Enter your query:")
        st.session_state["user_input"] = user_input

        self.table_previews()
        
        if user_input:
            st.session_state['progress'] = st.progress(0)
            output, generated_query = self.generate_query_response(user_input)
            st.session_state["generated_query"] = generated_query
            st.session_state["output"] = output
            st.write(output)
            #if sql is producted, display it and feed query to generate code
            if generated_query != "No SQL code found":
                with st.expander("SQL Query"):
                    st.code(generated_query)
                st.session_state['progress'].progress(.5)
                #generate the dataframe from the agent generated sql query
                df = self.generate_dataframe(st.session_state["generated_query"])
                st.session_state['progress'].progress(.6)
                st.session_state["data_frame"] = df
                #generate prefix and suffix
                primer_description, primer_code = generate_primer(df, "df")
                python_prompt_formatted = PYTHON_PROMPT.format(
                    user_input, generated_query
                )
                question_to_ask = format_primer(
                    primer_description, primer_code, python_prompt_formatted
                )
                #try to generate and execute the code. 
                try:
                    st.session_state['answer'] = generate_code(
                        question_to_ask, "gpt-4", api_key=self.openai_api_key
                    )
                    answer = st.session_state['answer']
                    st.session_state['progress'].progress(.8)
                    answer = primer_code + answer
                    try:
                        st.write(exec(answer))
                        st.session_state['progress'].progress(.9)
                    except Exception as exec_error:
                        st.write("Could not graph", exec_error)
                        

                    st.markdown("### The Code")
                    st.session_state['progress'].progress(1.0)
                    time.sleep(.4)
                    st.session_state['progress'].empty()
                    with st.expander("Code Written by Agent"):
                        st.code(answer)

                except Exception as general_exception:
                    st.write(general_exception)


if __name__ == "__main__":
    if "selected_model" not in st.session_state:
        st.session_state["selected_model"] = "gpt-3.5-turbo-16k-0613"

    SQL_AGENT = SQLAgent()
    SQL_AGENT.main()
