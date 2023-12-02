from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.llms.openai import OpenAI

from prompts import MSSQL_AGENT_PREFIX



class SQLAgent:
    def __init__(self, database_uri: str, llm: OpenAI, tables: list):
        self.database_uri = database_uri
        self.llm = llm
        self.tables = tables

    def create(self):
        db = SQLDatabase.from_uri(
            database_uri=self.database_uri,
            sample_rows_in_table_info=1,
            include_tables=self.tables,
            view_support=True,
        )

        toolkit = SQLDatabaseToolkit(db=db, llm=self.llm)
        return create_sql_agent(
            llm=self.llm,
            toolkit=toolkit,
            verbose=False,
            prefix=MSSQL_AGENT_PREFIX,
            #            format_instructions=SQL_AGENT_FORMAT_INSTRUCTIONS,
        )

    async def arun(self, query: str):
        agent = self.create()
        return agent.run(query)

    def run(self, query: str):
        agent = self.create()
        return agent.run(query)
