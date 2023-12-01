### Required setup ###
# pip install openai langchain sqlalchemy pyodbc
# sudo apt-get install unixodbc-dev #for DB connections
# install SQL ODBC (msodbcsql18):  https://learn.microsoft.com/en-us/sql/connect/odbc/linux-mac/installing-the-microsoft-odbc-driver-for-sql-server
# Information on SQL Agents https://python.langchain.com/en/latest/modules/agents/toolkits/examples/sql_database.html

from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.agents import AgentExecutor
from langchain.llms import OpenAI
from langchain.llms import LlamaCpp
import urllib

#### SQL CONNECTION ####
# Connection details
username = "sa"
password = "example"
server = "127.0.0.1"
database_name = "example"
driver = "ODBC+Driver+18+for+SQL+Server"


### LLM model ###
## Cloud
model = OpenAI()

## Edge
# model = LlamaCpp(model_path="./models/llama-7b-ggml-v2-q4_0.bin", verbose=True, n_threads=8, n_gpu_layers=26)
# model = LlamaCpp(model_path="./models/stable-vicuna-13B-ggml_q4_0.bin", verbose=True, n_threads=8, n_gpu_layers=10)
# model = LlamaCpp(model_path="./models/koala-7B.ggml.q4_0.bin", verbose=True, n_threads=8, n_gpu_layers=26)

# Create connection string
params = urllib.parse.quote_plus(f"DRIVER={{{driver}}};SERVER={server};DATABASE={database_name};UID={username};PWD={password};TrustServerCertificate=yes;Encrypt=yes")
connectionString=f"mssql+pyodbc:///?odbc_connect={params}"

# Create connection
db = SQLDatabase.from_uri(connectionString)
toolkit = SQLDatabaseToolkit(db=db)

agent_executor = create_sql_agent(
    llm=model,
    toolkit=toolkit,
    verbose=True
)

### Execute against the DB ###
agent_executor.run("Describe the Orders table")
agent_executor.run("How many Widgets were ordered")
