from langchain import SQLDatabase
from langchain.chat_models import ChatOpenAI
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from sqlalchemy import create_engine

OPENAI_API = "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"

engine = create_engine('sqlite:///events.db')
db = SQLDatabase(engine)

chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2, openai_api_key=OPENAI_API)

toolkit = SQLDatabaseToolkit(db=db, llm=chat)

db_agent_toolkit = create_sql_agent(
    llm=chat,
    toolkit=toolkit,
    max_iterations=3,
    verbose=True
)

# ТУТ НАПИШИ СВОГО ТГ БОТА

YOUT_REQUEST = "get my events"

LLM_RESPONCE = db_agent_toolkit(YOUT_REQUEST)

print(LLM_RESPONCE)
