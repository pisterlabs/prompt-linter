from langchain import OpenAI, SQLDatabase, SQLDatabaseChain
from langchain.chat_models import ChatOpenAI
from sqlalchemy.schema import *
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.agents import Tool

    
def get_tool():
    llm = ChatOpenAI(temperature=0.2, model_name="gpt-4")
    sql_db = SQLDatabase.from_uri(
        "<connection string>",
        sample_rows_in_table_info=1
    )
    sql_agent = create_sql_agent(
        llm=llm,
        toolkit=SQLDatabaseToolkit(db=sql_db, llm=llm),
        verbose=True,
        top_k=25,
        return_intermediate_steps=True
    )

    tool = Tool(
            name = "<Name this tool>",
            func=sql_agent.run,
            description="<Describe this tool>"
    )
    return tool