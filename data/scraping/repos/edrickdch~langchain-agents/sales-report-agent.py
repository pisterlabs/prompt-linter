from langchain.experimental.plan_and_execute import (
    PlanAndExecute,
    load_agent_executor,
    load_chat_planner,
)
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_toolkits import GmailToolkit, SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.tools.gmail.utils import build_resource_service, get_gmail_credentials
from dotenv import load_dotenv


def get_gmail_toolkit():
    credentials = get_gmail_credentials(
        token_file="token.json",
        client_secrets_file="credentials.json",
        scopes=["https://mail.google.com/"],
    )
    api_resource = build_resource_service(credentials=credentials)
    mail_toolkit = GmailToolkit(api_resource=api_resource)
    return mail_toolkit


def get_db_toolkit(llm):
    db = SQLDatabase.from_uri("sqlite:///data/main.db")
    db_toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    return db_toolkit


load_dotenv()

# Set up LLM
model = ChatOpenAI(temperature=0, model="gpt-4")

# Gmail Toolkit Setup
mail_toolkit = get_gmail_toolkit()

# Database Toolkit Setup
db_toolkit = get_db_toolkit(model)

# Tools
tools = [
    *mail_toolkit.get_tools(),
    *db_toolkit.get_tools(),
]

planner = load_chat_planner(model)
executor = load_agent_executor(model, tools, verbose=True)
agent = PlanAndExecute(planner=planner, executor=executor, verbose=True)

agent.run(
    "Query the database to find all sales person and sum the revenue they generated.\
        find the email of the head of sales in the database\
            and send them an email.\
                The content of the email should be the sum of the revenue made for each sales person."
)
