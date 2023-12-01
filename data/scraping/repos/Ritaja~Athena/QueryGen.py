# athe first thing before any other imports set lanchain traching
import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import openai
from helpers.SynapseSql import SynapseSql
from contextlib import contextmanager
from io import StringIO
from streamlit.runtime.scriptrunner.script_run_context import SCRIPT_RUN_CONTEXT_ATTR_NAME
from threading import current_thread
import streamlit as st
import sys


openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.getenv("OPENAI_API_BASE")
openai.api_type = os.getenv("OPENAI_API_TYPE")
openai.api_version = os.getenv("OPENAI_API_VERSION")
openai.engine = os.getenv("DEPLYMENT_ID")


sql_agent = SynapseSql(synapse_name = os.getenv("SYNAPSE_NAME"), 
                       synapse_sql_pool = os.getenv("DB_NAME"), 
                       synapse_user = os.getenv("SYNAPSE_USER"), 
                       synapse_password = os.getenv("SYNAPSE_PASSWORD"), 
                       llm_engine_name = os.getenv("DEPLOYMENT_ID"), 
                       topK=5)

@contextmanager
def st_redirect(src, dst):
    placeholder = st.empty()
    output_func = getattr(placeholder, dst)

    with StringIO() as buffer:
        old_write = src.write

        def new_write(b):
            if getattr(current_thread(), SCRIPT_RUN_CONTEXT_ATTR_NAME, None):
                buffer.write(b)
                output_func(buffer.getvalue())
            else:
                old_write(b)

        try:
            src.write = new_write
            yield
        finally:
            src.write = old_write


@contextmanager
def st_stdout(dst):
    with st_redirect(sys.stdout, dst):
        yield


@contextmanager
def st_stderr(dst):
    with st_redirect(sys.stderr, dst):
        yield

# st.title('The following Star schema table is populated in data lake')
# st.image(os.path.join(".", "tableStruct.jpg"))

st.title('Query in natural language')
query = st.text_input(
    'Query', 'How many tables are there?')


def get_sql_query(query):
    response = sql_agent.run(query)

    return response

st.title('OAI app response')
st.write(get_sql_query(query))

# with st_stdout("write"):
#     print(get_sql_query(query), file=sys.stdout)