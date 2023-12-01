from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.llms.openai import OpenAI
from langchain.agents import AgentExecutor
from langchain import OpenAI, SQLDatabase, SQLDatabaseChain
# from IPython.display import Markdown, HTML
from langchain.utilities import WikipediaAPIWrapper
from langchain.tools import BaseTool, StructuredTool, Tool, tool
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document

def get_sql_output(query):
    result = agent.run(query)
    return result

username  = 'root'                  # change it to a valid user on your db
password  = 'mango'  # fill in the appropriate password
uri       = f"mysql+pymysql://{username}:{password}@127.0.0.1/foodmart"

#postdb = SQLDatabase.from_uri("postgresql://abhi:mango@localhost:5432/abhi?sslmode=disable")
llm=OpenAI(temperature=0.1)
db = SQLDatabase.from_uri(uri)
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

agent = create_sql_agent(
    llm=OpenAI(temperature=1.5),
    toolkit=toolkit,
    verbose=True
)

# define wiki search tool
sql_tool = Tool.from_function(
        func=get_sql_output,
        name="sql",
        description="Use this tool only to search for any sql db related queries of foodmart.",
        return_direct=True
    )
