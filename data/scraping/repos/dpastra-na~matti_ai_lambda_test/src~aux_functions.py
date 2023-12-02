import os
from src.agents import SQLAgent
from langchain.chat_models import AzureChatOpenAI

def get_agent() -> SQLAgent:
    try:
        tables = [
            "v_detalles_alumnos",
            "v_transactions",
        ]
        llm = AzureChatOpenAI(
            openai_api_base=os.environ["OPENAI_API_BASE"],
            openai_api_key=os.environ["OPENAI_API_KEY"],
            openai_api_type = "azure",
            deployment_name="test",
            model_name="gpt-35-turbo",
            openai_api_version="2023-03-15-preview",
        )

        agent = SQLAgent(database_uri=os.environ['MSSQL_DB'] , llm=llm, tables=tables)
    except Exception as e:
        print(f'Error loading agent: {e}')
    return agent
