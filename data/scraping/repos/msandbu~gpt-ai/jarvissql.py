import os
from dotenv import load_dotenv
import openai
import langchain
import azure.cognitiveservices.speech as speechsdk
import elevenlabs
import json
import requests

from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.agents import AgentExecutor
from langchain.agents import create_sql_agent
from langchain import LLMMathChain, OpenAI, SQLDatabase, SerpAPIWrapper
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.streaming_stdout_final_only import (
    FinalStreamingStdOutCallbackHandler,
)

os.environ["OPENAI_API_KEY"] =""
os.environ["SQL_SERVER_USERNAME"] = "" 
os.environ["SQL_SERVER_ENDPOINT"] = ""
os.environ["SQL_SERVER_PASSWORD"] = ""  
os.environ["SQL_SERVER_DATABASE"] = ""
os.environ["SERPAPI_API_KEY"] =""

speech_key, service_region = "", "eastus"
speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config)

from sqlalchemy import create_engine
from sqlalchemy.engine.url import URL


db_config = {  
    'drivername': 'mssql+pyodbc',  
    'username': os.environ["SQL_SERVER_USERNAME"] + '@' + os.environ["SQL_SERVER_ENDPOINT"],  
    'password': os.environ["SQL_SERVER_PASSWORD"],  
    'host': os.environ["SQL_SERVER_ENDPOINT"],  
    'port': 1433,  
    'database': os.environ["SQL_SERVER_DATABASE"],  
    'query': {'driver': 'ODBC Driver 17 for SQL Server'}  
} 

from langchain.agents import create_sql_agent


llm = OpenAI(streaming=True,temperature=0)
search = SerpAPIWrapper()
db_url = URL.create(**db_config)
db = SQLDatabase.from_uri(db_url)
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
db_chain = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)
tools = [
    Tool(
        name = "Search",
        func=search.run,
        description="useful for when you need to answer questions about current events. You should ask targeted questions"
    ),
    Tool(
        name="FooBar-DB",
        func=db_chain.run,
        description="useful to answer questions about John in the database"
    )
]

while True:
    print("Talk now")
    result = speech_recognizer.recognize_once()
    print("Recognized: {}".format(result.text))
    message = format(result.text)
    agent = initialize_agent(tools, llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=True,)
    response = agent(
    {
        "input": result.text
    }
)
    response["output"]
    print(response["output"])
    audio_stream = elevenlabs.generate(text=response["output"],voice="Matthew", stream=True)
    output = elevenlabs.stream(audio_stream)

