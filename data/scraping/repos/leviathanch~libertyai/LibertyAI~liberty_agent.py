from langchain.llms import OpenAI
from langchain.agents import initialize_agent, Tool
from langchain.agents import load_tools
#from word2number import w2n
from datetime import date, datetime, timezone, timedelta

from langchain.vectorstores.pgvector import PGVector

from LibertyAI.liberty_llm import LibertyLLM
from LibertyAI.liberty_embedding import LibertyEmbeddings
from LibertyAI.liberty_config import get_configuration
from LibertyAI.liberty_weather_tool import get_current_weather

def current_time(arg):
    print(arg)
    now = datetime.now()
    today = date.today()
    ret = now.strftime("Date: %A, %B %dth, %Y. (%d/%m/%Y), Local Time: %H:%M %p")
    return ret

def get_date_time_tool():
    return Tool(
        name = "DateTime",
        func=current_time,
        description="When you are asked for the current time, you should use this tool."
    )

def get_vector_db():
    config = get_configuration()
    # DB Vectors in PostgreSQL:
    CONNECTION_STRING = PGVector.connection_string_from_db_params(
        driver="psycopg2",
        host=config.get('DATABASE', 'PGSQL_SERVER'),
        port=config.get('DATABASE', 'PGSQL_SERVER_PORT'),
        database=config.get('DATABASE', 'PGSQL_DATABASE'),
        user=config.get('DATABASE', 'PGSQL_USER'),
        password=config.get('DATABASE', 'PGSQL_PASSWORD'),
    )
    embeddings = LibertyEmbeddings(
        endpoint = "https://libergpt.univ.social/api/embedding"
    )
    db = PGVector(
        embedding_function = embeddings,
        connection_string = CONNECTION_STRING,
    )
    return db

def get_vector_db_tool():
    return Tool(
        name = "PGVector",
        func=get_vector_db().similarity_search_with_score,
        description="a QA database with books, it contains your opinions about things"
    )

def get_weather_tool():
    return Tool(
        name = "Weather",
        func=get_current_weather,
        description="a weather tool, useful for when you're asked about the current weather."
    )

def get_zero_shot_agent(llm):
    tools = []
    tools.append(get_weather_tool())
    #tools.append(get_vector_db_tool())
    #tools.append(get_date_time_tool())
    #tools += load_tools(
    #    ["searx-search"], searx_host="http://libergpt.univ.social/searx",
    #    llm = llm,
    #)
    mrkl = initialize_agent(
        tools,
        llm,
        agent="zero-shot-react-description",
        verbose=True
    )
    return mrkl


