import os
import streamlit as st

from dotenv import load_dotenv, find_dotenv
from langchain.agents import create_spark_sql_agent
from langchain.agents.agent_toolkits import SparkSQLToolkit
from langchain.chat_models import ChatOpenAI
from langchain.utilities.spark_sql import SparkSQL
from pyspark.sql import SparkSession

_ = load_dotenv(find_dotenv())

OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

spark = SparkSession.builder.getOrCreate()
schema = "titantic_schema"
spark.sql(f"CREATE DATABASE IF NOT EXISTS {schema}")
spark.sql(f"USE {schema}")
csv_file_path = "titanic.csv"
table = "titanic"
spark.read.csv(csv_file_path, header=True, inferSchema=True).write.saveAsTable(table)
spark.table(table).show()

# Note, you can also connect to Spark via Spark connect. For example:
# db = SparkSQL.from_uri("sc://localhost:15002", schema=schema)
spark_sql = SparkSQL(schema=schema)
llm = ChatOpenAI(temperature=0)
toolkit = SparkSQLToolkit(db=spark_sql, llm=llm)
agent_executor = create_spark_sql_agent(llm=llm, toolkit=toolkit, verbose=True)

st.title("Chat With Your Data in Spark! - Natural Language to SQL ..")
st.text_input("Please Enter Your Query in Plain Text ! ", key="query")
result = agent_executor.run(st.session_state.query)
st.write(result)

# "Describe the titanic table"
# "whats the square root of the average age?"
# "What's the name of the oldest survived passenger?"

