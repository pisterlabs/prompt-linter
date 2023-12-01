import pandas as pd
from langchain.chat_models import ChatOpenAI
from sqlalchemy import create_engine, inspect, text
import openai
from langchain.agents.agent_toolkits import create_python_agent
from langchain.tools.python.tool import PythonREPLTool
from langchain.callbacks import StreamlitCallbackHandler
import streamlit as st


# Query Database using GPT-3

# Let's build an application that allows us to query the database using GPT-3.
# In our application user will be able to connect to the database and ask questions in natural language.
# The application will use ChatGPT to convert the natural language question into SQL query and execute it.
# The application will return the results of the query to the user.
# Application will internally pass database schema information to ChatGPT so that it can use it to generate SQL queries.
# We will use the DVD rental database for this application.


# Function to classify the user input
def classify(user_input):
    """
    Function to classify the user input into one of the following categories: "query_db", "create_graph"
    :param user_input: User input
    :return: category
    """
    system_prompt = """
    A user is asking a question your task is to classify the question into one of the following categories: query_db, create_graph.\n
    query_db: choose this category if the user is asking a question that requires querying the database.\n
    create_graph: choose this category if the user is asking a question that requires creating a graph.\n
    Only output the category, do not add any additional information.
    For example: Show the top 5 R rated films
    query_db
    For example: can you create a bar chart showing the number of films by rating
    create_graph
    Remember to only output "query_db" or "create_graph" nothing else.
    """

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]
    )

    category = response.choices[0].message.content
    return category


# Function to query the database using GPT-3 and return the results
def query_db(nlp_query: str) -> pd.DataFrame:
    """
    Query the database using GPT-3 and return the results
    :param nlp_query: Natural language query
    :return: Pandas DataFrame containing the results of the query
    """
    # Create a prompt to query the database using GPT-3
    intro = """
    You are a data analyst working for a company. You have been asked to write SQL queries to answer \
    some business questions. You have been given a database schema and a list of questions. \
    You need to write SQL queries to answer these questions."""
    prompt = 'Given the database schema, write a SQL query that returns the following information: '
    prompt += nlp_query
    prompt += f'You only need to write SQL code, do not comment or explain code and do not add any additional info. \
    I need code only. Always use table name in column reference to avoid ambiguity. \
    SQL dialect is MySQL.\
    Only use columns and tables mentioned in the doc below. \n{doc}'

    # Query the database using GPT-3
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": intro},
            {"role": "user", "content": prompt}
        ]
    )

    # Extract the code from the response
    code = response.choices[0].message.content

    # Format the code
    pretty_code = '```sql\n' + code + '\n```'

    # Remove newlines from the code
    code = code.replace('\n', ' ')

    with engine.connect() as con:
        df = pd.read_sql_query(sql=text(code), con=engine)
        st.session_state['df'] = df
        st.session_state['pretty_code'] = pretty_code
    # Return the results
    with st.expander('Show results'):
        st.write(df)
        st.session_state.messages.append({"role": "assistant", "content": df})


# Function to create a graph using GPT-3
def create_graph(nlp_query: str):
    """
    Create a graph using python agent
    :param nlp_query: question with full context
    :return: None
    """
    llm = ChatOpenAI(temperature=0, openai_api_key=st.session_state["openai_api_key"])

    agent = create_python_agent(
        llm,
        tool=PythonREPLTool(),
        verbose=True
    )
    df = st.session_state['df']
    st_callback = StreamlitCallbackHandler(st.container())
    agent.run(nlp_query + f"using the following dataframe {df}" + "and save the figure in filename.png"
              , callbacks=[st_callback])

    # Get the filename.png from the current directory and display it
    filename = 'filename.png'
    st.image(filename)


def main(user_input):
    # Classify the user input
    category = classify(user_input)

    # Execute the query
    if category == 'query_db':
        query_db(user_input)
    elif category == 'create_graph':
        create_graph(user_input)
    else:
        print("Sorry I don't understand")


# Title
st.markdown("<h1 style='text-align: center; color: black;'>SQL GPT</h1>", unsafe_allow_html=True)
st.sidebar.title('SQL GPT')

# Ask user to enter open ai api key
st.session_state["openai_api_key"] = st.sidebar.text_input("Enter your open ai api key here", type="password")
openai.api_key = st.session_state["openai_api_key"]

db_products_dict = {
    'Default DB': ['mysql', 'mysql+pymysql'],
    'Postgres': ['postgres', 'postgresql+psycopg2'],
    'SQL Server': ['mssql', 'mssql+pyodbc'],
    'Oracle': ['oracle', 'oracle+cx_oracle'],
    'MySQL': ['mysql', 'mysql+pymysql'],

}

with st.sidebar:
    st.write('Pick your DB connection:')
    db_type = st.selectbox('DB connection', db_products_dict.keys())

    # If the user selects a database type apart from the default one, ask them to enter connection details
    if db_type != 'Default DB':
        db_host = st.text_input('Enter DB host')
        db_port = st.text_input('Enter DB port')
        db_user = st.text_input('Enter DB user')
        db_password = st.text_input('Enter DB password')
        db_name = st.text_input('Enter DB name')

    if st.button("Connect"):
        st.session_state['db_connection'] = True

if st.session_state.get('db_connection'):
    if db_type != 'Default DB':
        # check if the user has entered a database credentials
        if not db_host or not db_user or not db_password or not db_name or not db_port:
            st.error('Please enter the database credentials')
            st.stop()

    try:
        if db_type != 'Default DB':
            # Create a connection to the database
            engine = create_engine(
                f'{db_products_dict[db_type][1]}://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}')
        else:
            # Create a connection to the database
            engine = create_engine('sqlite:///dvd_rental.db')

        inspector = inspect(engine)

    except Exception as e:
        st.error(f'Could not connect to the database. Error: {e}')
        st.stop()

    # Get the table names
    table_names = inspector.get_table_names()

    # Get the column names
    column_names = {}

    for table_name in table_names:
        columns = inspector.get_columns(table_name)
        column_names[table_name] = [column['name'] for column in columns]

    # form an introspection doc
    doc = f"""
            AUTO-GENERATED DATABASE SCHEMA DOCUMENTATION\n
            Total Tables: {len(table_names)}
            """
    for table_name in table_names:
        doc += f"""
        ---
        Table Name: {table_name}\n
        Columns:
        """
        for column_name in column_names[table_name]:
            doc += f"""
            - {column_name}
            """

    with st.expander("See introspected DB structure"):
        st.write(doc)

    with st.expander('Latest dataframe'):
        if 'df' in st.session_state and not st.session_state['df'].empty:
            st.write(st.session_state['df'])
        else:
            st.write(None)

    with st.expander('Latest Executed SQL code'):
        st.code(st.session_state['pretty_code'] if st.session_state.get('pretty_code') else None)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about the database here... (e.g. How many customers are there?)"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("Thinking..."):
                main(prompt)
