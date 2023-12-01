from dotenv import load_dotenv
import streamlit as st
from langchain.llms import OpenAI
from langchain import OpenAI, SQLDatabase, SQLDatabaseChain
#import psycopg2
import openai
from services.GetEnvironmentVariables import GetEnvVariables

st.set_page_config(page_title="Ask your Database")
st.header("Ask your Database and get Insights!! 💬")

env_vars = GetEnvVariables()
OPENAI_API_KEY = env_vars.get_env_variable('openai_key')
openai.api_key=OPENAI_API_KEY

conn = ''

with st.form("my_form"):
   st.write("Inside the form")
   all_db_types = ["Mysql","Postgres","MongoDB"]
   db_type    = st.selectbox("Select DB Type",all_db_types)
   dbusername = st.text_input("select db username")
   dbpassword = st.text_input("select db password",type="password")
   dbname     = st.text_input("select db name")
   dbhost     = st.text_input("select db host")

   # Every form must have a submit button.
   submitted = st.form_submit_button("Submit")
   if submitted:
        if db_type=='Mysql':
            conn = "mysql+pymysql://"+dbusername+":"+dbpassword+"@"+dbhost+"/"+dbname

        if db_type=='Postgres':
            conn = "postgresql+psycopg2://"+dbusername+":"+dbpassword+"@"+dbhost+"/"+dbname
            
        st.write(conn)
        
        if conn!='':
            db =  SQLDatabase.from_uri(
                conn,
                )

            # setup llm
            llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)


            # Setup the database chain
            db_chain = SQLDatabaseChain(llm=llm, database=db, verbose=True)

            user_question = st.text_input("write a query:")
            if user_question!='':
                st.write(db_chain.run(user_question))

   

#st.write(db)
