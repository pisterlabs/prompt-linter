from django.db import connections
import openai
from django.conf import settings
from sqlalchemy import URL
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.agents import create_sql_agent
from langchain.llms import AzureOpenAI, OpenAI

# Connect to database


class Connect:
    def __init__(self, database=None):
        self.database = "default" if database is None else database
        self.query = None
        self.starter = None

    def cursor(self, getter=None):
        if getter:
            return connections.databases[self.database][getter]
        return connections.databases[self.database]

    def engage(self):
        engine = self.cursor("ENGINE")

        if engine == "django.db.backends.mysql":
            self.starter = "mysql+pymysql"
            self.query = self.cursor("OPTIONS")

        elif engine == "django.db.backends.postgresql":
            self.starter = "postgres"
            self.query = self.cursor("OPTIONS")

        elif engine == "django.db.backends.oracle":
            self.starter = "oracle+cx_oracle"
            self.query = self.cursor("OPTIONS")

        elif engine == "mssql":
            self.starter = "mssql+pyodbc"
            query = self.cursor("OPTIONS")
            new_query = {}
            if "extra_params" in query.keys():
                key_value_pairs = query["extra_params"].strip(";").split(";")
                new_query.update({kv.split("=")[0]: kv.split("=")[1] for kv in key_value_pairs})
            query.pop("extra_params")
            new_query.update(query)
            self.query = new_query

        else:
            raise Exception("The database engine {} is not yet supported.".format(engine))

    def connection_uri(self):
        return URL.create(
                self.starter,
                username=self.cursor("USER"),
                password=self.cursor("PASSWORD"),
                host=self.cursor("HOST"),
                port=self.cursor("PORT"),
                database=self.cursor("NAME"),
                query=self.query
        )

    def data(self):
        return SQLDatabase.from_uri(self.connection_uri())


class DjangoAI:
    def __init__(self, db):
        self.db = db
        if settings.OPENAI_API_TYPE == "azure":
            openai.api_key = settings.OPENAI_API_KEY
            openai.api_base = settings.OPENAI_API_BASE
            openai.api_type = settings.OPENAI_API_TYPE
            openai.api_version = settings.OPENAI_API_VERSION
            self.llm = AzureOpenAI(deployment_name=settings.OPENAI_DEPLOYMENT_NAME)
        else:
            self.llm = OpenAI(openai_api_key=settings.OPENAI_API_KEY)
        self.toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)
        self.agent = create_sql_agent(llm=self.llm, toolkit=self.toolkit, verbose=True)

    def think(self, message):
        return self.agent.run(message)
