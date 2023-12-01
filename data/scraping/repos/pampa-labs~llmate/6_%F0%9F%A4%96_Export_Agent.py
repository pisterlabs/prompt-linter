import streamlit as st
from langchain.agents.agent_toolkits.sql.prompt import SQL_PREFIX, SQL_FUNCTIONS_SUFFIX


import llmate_config
llmate_config.general_config()

if ('openai_api_key' not in st.session_state) or (st.session_state['openai_api_key'] == ''):
    st.error('Please load OpenAI API KEY and connect to a database', icon='🚨')
else:

    st.info("To recreate this Agent in your solution, copy and paste the code below:")



    # First we define the logic to see wether we need to specify params or not:
    changing_tables = (st.session_state['include_tables'] != st.session_state['table_names'])
    changing_prefix = (st.session_state['sql_agent_prefix'] != SQL_PREFIX)
    changing_suffix = (st.session_state['sql_agent_suffix'] != SQL_FUNCTIONS_SUFFIX)
    adding_few_shots = ('few_shots' in st.session_state)

    extra_imports = '''from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.agents.agent_types import AgentType
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain.vectorstores import FAISS

import json
'''

    code = f'''
from langchain.chat_models import ChatOpenAI
from langchain.utilities import SQLDatabase
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType
from langchain.agents import create_sql_agent
{extra_imports if adding_few_shots else ''}

llm = ChatOpenAI(temperature=0, verbose=True)

'''

    if changing_tables:
        code += f"# List containing the tables to include \ninclude_tables = {st.session_state['include_tables']}\n\n"

    if 'custom_table_info' in st.session_state:
        code += f'# Custom table info \ncustom_table_info = {st.session_state["custom_table_info"]}\n\n'

    code += '''
# Replace with your database URI
database_uri = "sqlite:///your-database-uri.db"

# Load DB
sql_db = SQLDatabase.from_uri(database_uri,'''

    if changing_tables:
        code += '''
                            include_tables=include_tables,'''


    code += f'''
                            custom_table_info=custom_table_info
                            )


sql_toolkit = SQLDatabaseToolkit(db=sql_db, llm=llm)

    '''

    if adding_few_shots:
        code += '''
# Including the few shot examples tool

with open('few_shots.json', 'r') as json_file: # replace with your few shots path
    few_shots = json.load(json_file)

embeddings = OpenAIEmbeddings()

# Create vectorstore docs
few_shot_docs = [
        Document(
            page_content=example["question"],
            metadata={"sql_query": example["sql_query"]},
            )
        for example in few_shots
    ]
vector_db = FAISS.from_documents(few_shot_docs, embeddings)
retriever = vector_db.as_retriever()
description = """
This tool will help you understand similar examples to adapt them to the user question.
Input to this tool should be the user question.
"""

retriever_tool = create_retriever_tool(
    retriever, name="sql_get_similar_examples", description=description
)
'''

    if changing_prefix:
        code += f'''

agent_prefix = """
{st.session_state['sql_agent_prefix']}
"""
        '''
    if changing_suffix:
        code += f'''
agent_suffix = """
{st.session_state['sql_agent_suffix']}
"""
        '''

    code += f'''
agent = create_sql_agent(llm = llm,
                        toolkit=sql_toolkit,
                        verbose=True,
                        agent_type=AgentType.OPENAI_FUNCTIONS,'''

    if adding_few_shots:
        code += '''
                        extra_tools=[retriever_tool],'''
    
    if changing_prefix:
        code += '''
                        prefix=agent_prefix,'''

    if changing_suffix:
        code += '''
                        suffix=agent_suffix,'''
    code += '''
                        )
    '''
    st.code(code, language='python')