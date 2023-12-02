import urllib, os
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.llms.openai import AzureOpenAI

class SynapseSql:
    def __init__(self, synapse_name, synapse_sql_pool, synapse_user, synapse_password, llm_engine_name, topK=10) -> None:
        

        synapse_connect_str = "Driver={{ODBC Driver 18 for SQL Server}};Server=tcp:{}.sql.azuresynapse.net,1433;" \
                      "Database={};Uid={};Pwd={};Encrypt=yes;TrustServerCertificate=no;" \
                      "Connection Timeout=30;".format(synapse_name, synapse_sql_pool, synapse_user, synapse_password)

        params = urllib.parse.quote_plus(synapse_connect_str)
        self.conn_str = 'mssql+pyodbc:///?odbc_connect={}'.format(params)

        db = SQLDatabase.from_uri(self.conn_str)
        llm = AzureOpenAI(temperature=0,  deployment_name=llm_engine_name)
        self.toolkit = SQLDatabaseToolkit(db=db,llm=llm)

        self.SQL_PREFIX = """You are an agent designed to interact with SQL database systems.
        Given an input question, create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
        Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results using SELECT TOP in SQL Server syntax.
        You can order the results by a relevant column to return the most interesting examples in the database.
        Never query for all the columns from a specific table, only ask for a the few relevant columns given the question.
        You have access to tools for interacting with the database.
        Only use the below tools. Only use the information returned by the below tools to construct your final answer.
        You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

        DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

        If the question does not seem related to the database, just return "I don't know" as the answer.
        """

        self.agent_executor = create_sql_agent(
                llm=llm,
                toolkit=self.toolkit,
                verbose=True,
                prefix=self.SQL_PREFIX, 
                topK = topK
            )
        
    def run(self, text: str):
        return self.agent_executor.run(text)
