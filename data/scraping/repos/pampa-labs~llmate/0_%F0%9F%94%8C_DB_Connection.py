import streamlit as st
from langchain import OpenAI, SQLDatabase
# from langchain_experimental.sql import SQLDatabaseChain
from streamlit_extras.grid import grid
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import create_sql_agent
from langchain.agents.agent_types import AgentType
import pymysql
pymysql.install_as_MySQLdb()



import llmate_config

llmate_config.general_config()
llmate_config.init_session_state()

# if 'db_uri'not in st.session_state: 
#     st.session_state['db_uri'] = ''
connected = False 

if (st.session_state['openai_api_key'] != ''): 

    st.header('LLMate 🧉')
    dialects = ['sqlite', 'mysql']

    st.session_state.dialect = st.selectbox('Select a database dialect:', dialects)

    if st.session_state.dialect == 'sqlite':

        my_grid = grid(1, vertical_align="bottom")
        st.session_state.database_path = st.text_input('Path to SQLite database file', value= 'example\Example_Chinook.db')
    else:
        my_grid = grid(2,2,1, vertical_align="bottom")
        st.session_state.username = my_grid.text_input('username')
        st.session_state.password = my_grid.text_input('password', type= "password")
        st.session_state.host = my_grid.text_input('host', "localhost") 
        st.session_state.port = my_grid.text_input('port', "3306")  # default port for MySQL
        st.session_state.database_name = my_grid.text_input('db name')



    col1, col2, col3 = st.columns(3)

    with col1:
        st.write(' ')

    with col2:
        if st.button('Connect to db', use_container_width=True):
            if (st.session_state.dialect == 'sqlite') & (st.session_state['database_path'] != ''):
                try:
                    st.session_state['db_uri'] = f"sqlite:///{st.session_state.database_path}"
                    st.session_state['sql_db'] = SQLDatabase.from_uri(st.session_state['db_uri'])
                    st.session_state['db_conn'] = st.session_state.database_path
                    st.success(f"Connected to {st.session_state['db_conn']}")
                    connected = True
                except:
                    st.error('Connection failed')

            if st.session_state.dialect == 'mysql':
                try: 
                    st.session_state['db_uri'] = f"{st.session_state.dialect}+pymysql://{st.session_state.username}:{st.session_state.password}@{st.session_state.host}:{st.session_state.port}/{st.session_state.database_name}"
                    print(st.session_state['db_uri'])
                    st.session_state['sql_db'] = SQLDatabase.from_uri(st.session_state['db_uri'])
                    st.session_state['db_conn'] = st.session_state.database_name
                    st.success(f"Connected to {st.session_state['db_conn']}")
                    connected = True
                except:
                    st.error('Connection failed')

            if connected:
                st.session_state['include_tables'] = st.session_state['sql_db'].get_table_names()
                st.session_state['table_names'] = st.session_state['sql_db'].get_table_names()
                tables_createtable_statement = st.session_state['sql_db'].get_table_info().split("CREATE TABLE")[1:]
                custom_table_info = {}

                for i in range(len(tables_createtable_statement)):
                    custom_table_info[st.session_state['table_names'][i]] = "CREATE TABLE " + tables_createtable_statement[i]
                st.session_state['custom_table_info'] = custom_table_info

                st.session_state['sql_toolkit'] = SQLDatabaseToolkit(db=st.session_state['sql_db'],
                                                                    llm=st.session_state['llm'],
                                                                    custom_table_info=st.session_state['custom_table_info']
                                                                    )
                st.session_state['sql_agent'] = create_sql_agent(
                    llm = st.session_state['llm'],
                    toolkit=st.session_state['sql_toolkit'],
                    verbose=True,
                    agent_type=AgentType.OPENAI_FUNCTIONS,
                    prefix=st.session_state['sql_agent_prefix'],
                    suffix=st.session_state['sql_agent_suffix']
                )
            

else:
    st.error('Please load OpenAI API KEY', icon='🚨')

            

