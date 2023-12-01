import streamlit as st
import openai
from langchain.utilities import SQLDatabase
from langchain.chat_models import ChatOpenAI
from langchain.chains import create_sql_query_chain
from langchain.sql_database import SQLDatabase
import pandas as pd
import sqlparse
import textwrap
import re
from  sqlalchemy import create_engine
from langchain.prompts.prompt import PromptTemplate
from sqlalchemy import text
from dotenv import load_dotenv
from langchain import LLMChain

load_dotenv()

import langchain
langchain.debug=False

llm = ChatOpenAI(temperature=0)


# con = sqlite3.connect("chinook.db")
region = 'us-east-1'
glue_database_name='chinook'
glue_databucket_name='aws-athena-query-results-xxxxxx-us-east-1'
connathena=f"athena.{region}.amazonaws.com" 
portathena='443' #Update, if port is different
schemaathena=glue_database_name #from cfn params
s3stagingathena=f's3://{glue_databucket_name}/athenaresults/'#from cfn params
wkgrpathena='primary'#Update, if workgroup is different
# tablesathena=['dataset']#[<tabe name>]
##  Create the athena connection string
connection_string = f"awsathena+rest://@{connathena}:{portathena}/{schemaathena}?s3_staging_dir={s3stagingathena}/&work_group={wkgrpathena}"
##  Create the athena  SQLAlchemy engine
engine_athena = create_engine(connection_string, echo=False)
# db = SQLDatabase(engine_athena)



# db = SQLDatabase.from_uri("sqlite:///Chinook.db")
db = SQLDatabase(engine_athena, sample_rows_in_table_info=0, custom_table_info={})





st.title("🦜🔗 Generative SQL with Athena")

st.write('The Chinook sample database for a digital media store can be used to explore')
with st.expander("See table info"):
    st.image(image='chinook_schema.jpg', caption='DB schema')

PROMPT_SUFFIX = """Only use the following tables:
{table_info}

Question: {input}"""

_DEFAULT_TEMPLATE = """Given an input question, first create a syntactically correct Presto query to run, then look at the results of the query and return the answer. Unless the user specifies in his question a specific number of examples he wishes to obtain, always limit your query to at most {top_k} results. You can order the results by a relevant column to return the most interesting examples in the database.

Never query for all the columns from a specific table, only ask for a the few relevant columns given the question.

Pay attention to use only the column names that you can see in the schema description. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.

Make sure selecting the columns only which is in GROUP BY. 
If the group column is a aggregated column, make sure the same aggregation calculation is also used in GROUP BY.

If you use string indicating a date, add date before the string. For example, date '2012-01-01'. Other than this, avoid to use date and time functions and Operator which may not be supported in Presto query.

Rename the columns to the best of answering the question.

Review the answer and improve before giving the answer.

If you think the question is not related to any tables in the database, just reply 'Sorry, it seems not related to the data'.

Use the following format:

Question: Question here
SQLQuery: SQL Query to run
SQLResult: Result of the SQLQuery
Answer: Final answer here

"""


CUSTOM_PROMPT = PromptTemplate(
    input_variables=["input", "table_info",  "top_k"],
    template=_DEFAULT_TEMPLATE + PROMPT_SUFFIX
)

chain = create_sql_query_chain(llm, db, prompt=CUSTOM_PROMPT)


sql_correct_prompt_template = '''Given an SQL script and a error message. create a syntactically correct Presto query to run, then look at the results of the query and return the answer. Let's think step by step. Only return the "improved sql" script as output.


SQL: {sql}
error: {error}

<corrected sql as output>





'''
sql_correct_prompt = PromptTemplate(
    input_variables=["sql", "error"],
    template=sql_correct_prompt_template
)

sql_correct_chain = LLMChain(llm=llm, prompt=sql_correct_prompt)



# Set OpenAI API key from Streamlit secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Set a default model
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])



# Accept user input
if prompt := st.chat_input("What is up?"):
    st.empty()
    # Add user message to chat history
    # st.session_state.messages.append({"role": "user", "content": prompt})

    question = prompt
    sql_response = ''
    data_response = ''

    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Display assistant response in chat message container

        # for response in chain.invoke({"question":prompt}):
        #     full_response += response.choices[0].delta.get("content", "")
        #     message_placeholder.markdown(full_response + "▌")

    sql_response = chain.invoke({"question":prompt})

    if(sql_response == 'Sorry, it seems not related to the data.'):
        st.chat_message("assistant").text('Sorry, it seems not related to the data.')
        st.stop()
    
    
    # Detect the language of the code
    sql_keywords_regex = r'^(Sorry|WITH|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER)'
    match = re.search(sql_keywords_regex, sql_response.strip(), re.IGNORECASE)
    if not match:
        formatted_code = sqlparse.format(sql_response, reindent=True)
        formatted_code = textwrap.indent(formatted_code, " " * 4)

        st.chat_message("system").code(formatted_code)
    else:
        st.chat_message("system").markdown(sql_response)
        st.chat_message("system").error("SQL execution is not allowed")
        st.stop()
    # st.session_state.messages.append({"role": "system", "content": sql_response})

    # with st.chat_message("assistant"):
    #     message_placeholder = st.empty()
    full_response = ''
    # data_response = db.run(sql_response)
    
    retry = 0
    while(True):
        try:
            with engine_athena.connect() as conn:
                df = pd.read_sql_query(text(sql_response), con=conn)    
                st.chat_message("assistant").dataframe(df)               
        except Exception as e:
            st.chat_message("system").error(e)
            #TODO improve the SQL based on error
            improved_sql = sql_correct_chain.run({"sql": sql_response, "error":e})
            st.chat_message("system").code(improved_sql)
            sql_response=improved_sql
            retry +=1
            if retry > 5:
                st.chat_message("system").error("Still failed after retries")
                break
            else:
                continue
        break

    

        

