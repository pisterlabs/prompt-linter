import streamlit as st
from langchain.utilities import SQLDatabase
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import create_sql_agent
from langchain.agents.agent_types import AgentType

from utils import update_agent


import llmate_config
llmate_config.general_config()

if ('openai_api_key' not in st.session_state) or (st.session_state['openai_api_key'] == ''):
    st.error('Please load OpenAI API KEY and connect to a database', icon='🚨')
else:
    if 'custom_table_info' not in st.session_state:
            tables_createtable_statement = st.session_state['sql_db'].get_table_info().split("CREATE TABLE")[1:]
            custom_table_info = {}

            for i in range(len(tables_createtable_statement)):
                custom_table_info[st.session_state['table_names'][i]] = "CREATE TABLE " + tables_createtable_statement[i]
            st.session_state['custom_table_info'] = custom_table_info

    def update_db_params():
        if st.session_state['selected_tables']:
            st.session_state['include_tables'] = st.session_state['selected_tables']
            st.session_state['sql_db'] = SQLDatabase.from_uri(st.session_state['db_uri'],
                                                            include_tables=st.session_state['include_tables'],
                                                            # sample_rows_in_table_info=st.session_state['sample_rows_in_table_info']
                                                            )
            
            st.session_state['sql_toolkit'] = SQLDatabaseToolkit(db=st.session_state['sql_db'],
                                                                llm=st.session_state['llm'],
                                                                custom_table_info=st.session_state['custom_table_info']
                                                                )
            
            update_agent()
            
            tables_createtable_statement = st.session_state['sql_db'].get_table_info().split("CREATE TABLE")[1:]
            custom_table_info = {}

            for i in range(len(tables_createtable_statement)):
                custom_table_info[st.session_state['include_tables'][i]] = "CREATE TABLE " + tables_createtable_statement[i]
            
            st.session_state['custom_table_info'] = custom_table_info

    def update_table_info(table_id=None):
        # No need for edited_info_dict as we update directly in session_state
        edited_info = st.session_state.get(f'table_info_editor_{table_id}', None)
        if edited_info:
            print("Info editada:", edited_info)
            st.session_state['custom_table_info'][st.session_state['include_tables'][table_id]] = edited_info
            print("Guardando edited key:", st.session_state['include_tables'][table_id])
            print("Info guardada:", st.session_state['custom_table_info'][st.session_state['include_tables'][table_id]])

        st.session_state['sql_db'] = SQLDatabase.from_uri(st.session_state['db_uri'],
                                                        include_tables=st.session_state['include_tables'], 
                                                        custom_table_info=st.session_state['custom_table_info'])
        
        st.session_state['sql_toolkit'] = SQLDatabaseToolkit(db=st.session_state['sql_db'],
                                                            llm=st.session_state['llm'])
        
        print("Info SQL Toolkit:", st.session_state['sql_toolkit'].get_tools()[0].db.table_info)
        
        st.session_state['sql_agent'] = create_sql_agent(
            llm=st.session_state['llm'],
            toolkit=st.session_state['sql_toolkit'],
            verbose=True,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            prefix=st.session_state['sql_agent_prefix'],
            suffix=st.session_state['sql_agent_suffix']
        )

        if table_id is not None:
            st.toast(f"Updated **{st.session_state['include_tables'][table_id]}** table info", icon='✅')


    st.subheader('Edit Database Information')
    st.markdown(
    """
    **Why change the table information before including it in the prompt? 🤔**
    
    Well, think of it like handing in a well-prepared report instead of just scribbled notes. 
    By choosing which tables to include and editing the table information, you're essentially making the Agent's 'reading' experience smoother and more focused.

    First, select the tables you want. After that, you can dive deeper and edit their specific information.    
    """
    )

    if 'checkbox_states' not in st.session_state:
        st.session_state.checkbox_states = {table: True for table in st.session_state['table_names']}

    with st.expander("1. Tables selection"):
        selected_tables = []
        for table in st.session_state['table_names']:
            if st.checkbox(table, value=st.session_state.checkbox_states[table], key=table):
                selected_tables.append(table)
            else:
                st.session_state.checkbox_states[table] = False

        st.session_state['selected_tables'] = selected_tables

        if st.button('Update tables'):
            update_db_params()
            st.success(f"Tables updated: {selected_tables}")


    with st.expander("2. Edit Create Statements"):
        if st.session_state['selected_tables']:
            tabs = st.tabs(st.session_state['include_tables'])
            i = 0
            for tab in tabs:
                with tab:
                    st.text_area("`Table info`",
                                value=st.session_state['custom_table_info'][st.session_state['include_tables'][i]],
                                height=500,
                                on_change=update_table_info,
                                key=f'table_info_editor_{i}',
                                args=[i],
                                label_visibility='collapsed')
                    i += 1
        else:
            st.warning("Selecciona al menos una tabla en el paso anterior.")

    with st.expander("View table info to be used by the Agent"):
        current_toolkit_info = st.session_state['sql_toolkit'].get_tools()[0].db.table_info
        st.text(current_toolkit_info)
