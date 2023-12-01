from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.llms.openai import OpenAI
from langchain.agents import AgentExecutor
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import streamlit as st

from utils.sidebar import sidebar
from utils.streaming import StreamHandler

load_dotenv()

# Update the database URI to point to your enhanced_database.db
db_uri =  "sqlite:///D:/University/challenge/Marketing List Chatbot/Solution/enhanced_database.db"
db = SQLDatabase.from_uri(db_uri)

toolkit = SQLDatabaseToolkit(db=db, llm=OpenAI(temperature=0))

# Streamlit App
sidebar()

st.title("Customer Insights App")
st.write("Welcome to the Customer Insights App. Enter your query below:")

query = st.text_area("Enter your query here:")

placeholder = st.empty()
placeholder.write("*[Agent Chatter will appear here]*")
st_cb = StreamHandler(placeholder)
chat = ChatOpenAI(streaming=True, callbacks=[st_cb])

agent_executor = create_sql_agent(
    llm=chat,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)

if st.button("Get Insights"):
    response = agent_executor.run(query)
    st.subheader("Response:")
    st.markdown(response)
  #  print(response['Action Input'])
