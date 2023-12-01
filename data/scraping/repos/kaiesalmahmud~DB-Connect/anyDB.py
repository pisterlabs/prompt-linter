import os
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.llms.openai import OpenAI
from langchain.agents import AgentExecutor
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI

import streamlit as st
from io import StringIO

API_KEY = open('key.txt', 'r').read().strip()
os.environ["OPENAI_API_KEY"] = API_KEY

import openai
openai.api_key = API_KEY

from dotenv import load_dotenv
load_dotenv()


# SQL_PREFIX = """You are an agent designed to interact with a SQL database.
# Given an input question, create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
# Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results.
# You can order the results by a relevant column to return the most interesting examples in the database.
# Never query for all the columns from a specific table, only ask for the relevant columns given the question.
# You have access to tools for interacting with the database.
# Only use the below tools. Only use the information returned by the below tools to construct your final answer.
# You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

# DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

# If the question does not seem related to the database, just return "I don't know" as the answer.

# """

SQL_FUNCTIONS_SUFFIX = """I should look at the tables in the database to see what I can query.  Then I should query the schema of the most relevant tables."""

FORMAT_INSTRUCTIONS = """Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question
"""

def get_response(input_text):
    response = agent_executor(input_text)

    # print(response['intermediate_steps'][1][0].tool)
    # print(response['intermediate_steps'][-1][0].tool)
    # print(response['output'])


    if response['intermediate_steps'][1][0].tool == 'sql_db_schema':
        schema = response['intermediate_steps'][1][1]
    else: schema = None

    if response['intermediate_steps'][-1][0].tool == 'sql_db_query':
        query = response['intermediate_steps'][-1][0].tool_input
        query_output = response['intermediate_steps'][-1][1]
    else: query, query_output = None, None

    answer = response['output']

    return schema, query, query_output, answer

def explain(query, schema, query_output):

    message_history = [{"role": "user", "content": f"""You are a SQL query explainer bot. That means you will explain the logic of a SQL query. 
                    There is a postgreSQL database table with the following table:

                    {schema}                   
                    
                    A SQL query is executed on the table and it returns the following result:

                    {query_output}

                    I will give you the SQL query executed to get the result and you will explain the logic executed in the query.
                    Make the explanation brief and simple. It will be used as the explanation of the results. Do not mention the query itself.
                    No need to explain the total query. Just explain the logic of the query.
                    Reply only with the explaination to further input. If you understand, say OK."""},
                   {"role": "assistant", "content": f"OK"}]

    message_history.append({"role": "user", "content": query})

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=message_history,
    )

    explaination = completion.choices[0].message.content
    return explaination

# host="localhost"
# port="5432"
# database="ReportDB"
# user="postgres"
# password="postgres"

# host="rain.db.elephantsql.com"
# port="5432"
# database="wblrcksm"
# user="wblrcksm"
# password="gElzAF-zRYJO-DNtPUudJ7pV0C4E6qMv"

# Create the sidebar for DB connection parameters

# Create the main panel
st.title("anyDB :sparkles:")
st.subheader("Connect your database and ask questions!")

host="localhost"
port="5432"
database="ReportDB"
username="postgres"
password="postgres"

with st.form("Connect Your Database"):

    # st.markdown("#### Connect Your Database")
    host = st.text_input("Host", placeholder=host)
    port = st.text_input("Port", placeholder=port)
    username = st.text_input("Username", placeholder=username)
    password = st.text_input("Password", placeholder=password)
    database = st.text_input("Database", placeholder=database)

    submit_button = st.checkbox("Connect")

    description = ""

    uploaded_file = st.file_uploader("Upload Database Documentation")
    if uploaded_file is not None:
        # To read file as bytes:
        bytes_data = uploaded_file.getvalue()
        # st.write(bytes_data)

        # To convert to a string based IO:
        description = str(StringIO(uploaded_file.getvalue().decode("utf-8")))

    submit = st.form_submit_button("Process")

# print(description)

SQL_PREFIX = """You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.
You have access to tools for interacting with the database.
Only use the below tools. Only use the information returned by the below tools to construct your final answer.
You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

If the question does not seem related to the database, just return "I don't know" as the answer.

Here are some information about the database:

"""

SQL_PREFIX_update = SQL_PREFIX + "\n" + description

if submit_button:

    db = SQLDatabase.from_uri(f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{database}")

    llm = ChatOpenAI(model_name="gpt-4", temperature=0)

    toolkit = SQLDatabaseToolkit(db=db, llm=llm)

    agent_executor = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        prefix=SQL_PREFIX_update,
        suffix=SQL_FUNCTIONS_SUFFIX,
        format_instructions=FORMAT_INSTRUCTIONS,
        agent_executor_kwargs = {'return_intermediate_steps': True}
    )

    st.sidebar.success("Connected to the database!")

# Get the user's natural question input
question = st.text_input(":blue[Ask a question:]", placeholder="Enter your question.")

# Create a submit button for executing the query
query_button = st.button("Submit")

# Execute the query when the submit button is clicked
if query_button:

    if not submit_button:
        st.warning(":wave: Please connect to the database first.")
        st.stop()

    # Display the results as a dataframe
    # Execute the query and get the results as a dataframe
    try:
        with st.spinner('Calculating...'):
            print("\nQuestion: " + str(question))
            # print(str(question))
            schema, query, query_output, answer = get_response(question)

            if query:
                explaination = explain(query, schema, query_output)
            else: explaination = None

            # explaination = explain(query, schema, query_output)

        # if query:
        #     print("\nExplaination: " + str(explaination))

        print("\nExplaination: " + str(explaination))

        st.subheader("Answer :robot_face:")
        st.write(answer)

        try:
            if query:

                st.divider()
                # st.caption("Query:")
                # st.caption(query)

                st.caption("Explaination:")
                st.caption(explaination)

                st.divider()
        except Exception as e:
            print(e)

        st.info(":coffee: _Did that answer your question? If not, try to be more specific._")
    except Exception as e:
        print(e)
        st.warning(":wave: Please enter a valid question. Try to be as specific as possible.")

