import os
import openai
from dotenv import load_dotenv
load_dotenv()

## OPEN AI API KEY
openai_key = os.getenv('OPENAI_API_KEY')
openai.api_key = openai_key

from dbconnector import DBcomm
SQL_URI = DBcomm.connection_uri
SQL_ENGINE = DBcomm.sql_engine

from langchain import OpenAI, SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.memory import ConversationBufferMemory
from langchain.agents import create_sql_agent 
from langchain.agents import AgentType
from langchain.agents.agent_toolkits import SQLDatabaseToolkit 
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
from langchain.schema import BaseOutputParser
from langchain.agents import Tool
from langchain.agents import initialize_agent

from llama_index.langchain_helpers.agents import LlamaToolkit, create_llama_chat_agent, IndexToolConfig
from llama_index import (LLMPredictor, ServiceContext, SimpleDirectoryReader,
                         SQLDatabase, StorageContext, VectorStoreIndex,
                         set_global_service_context)

from llama_index.objects import (ObjectIndex, SQLTableNodeMapping,
                                 SQLTableSchema)

# DB Interface library
from sqlalchemy import (Column, Integer, MetaData, String, Table, column,
                        create_engine, select, inspect)






class CommaSeparatedListOutputParser(BaseOutputParser):
    """Parse the output of an LLM call to a comma-separated list."""
    def parse(self, text: str):
        """Parse the output of an LLM call."""
        return text.strip().split(", ")

template = """You are a helpful assistant who extracts a list of SQL Tables given a user query and a DB schema.
A user will pass in a category, and you should generate 5 objects in that category in a comma separated list.
ONLY return a comma separated list, and nothing more."""
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_template = "{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
chain = LLMChain(
    llm=ChatOpenAI(),
    prompt=chat_prompt,
    output_parser=CommaSeparatedListOutputParser()
)
chain.run("colors")
# >> ['red', 'blue', 'green', 'yellow', 'orange']


class KwairyLite() :
    ######## PRE-INITIALIZATION ROUTINES
    def create_tableschema_index (self) :
        inspector = inspect(DBcomm.sql_engine)
        self.sql_table_names = inspector.get_table_names()
        self.indices_created = False
        self.sqldb, self.schemaindex = None, None
        #### SQL DB index
        # load all table definitions as indexes for retrieval later
        print("Loading table schema as object index")
        metadata_obj = MetaData()
        metadata_obj.reflect(DBcomm.sql_engine)
        sql_database = SQLDatabase(DBcomm.sql_engine)
        table_node_mapping = SQLTableNodeMapping(sql_database)
        table_schema_objs = []
        for table_name in metadata_obj.tables.keys():
            table_schema_objs.append(SQLTableSchema(table_name=table_name))
        # Dump the table schema information into a vector index. The vector index is stored within the context builder for future use.
        tableschema_index = ObjectIndex.from_objects(
            table_schema_objs,
            table_node_mapping,
            VectorStoreIndex,
        )
        self.sqldb, self.schemaindex = sql_database, tableschema_index
        print("Done loading schema...")

    ######## INITIALIZATION ROUTINES
    def __init__(self) :
        print("Initiating Kwairy")
        self.create_tableschema_index()
    
    def _process_user_query_(self, query : str) :
        pass

    def getSQLagent(self) :
        from langchain.prompts.prompt import PromptTemplate

        _DEFAULT_TEMPLATE = """ You are Kwairy, an experienced databases expert.
                                Given a user input, extract relevant SQL table schemas from input queries using the provided tool. 
                                Ensure you understand the user's intention. You can ask questions to the user.
                                For example, if the user mentions 'worker', they might mean 'employee' table in this imaginary example. 
                                Clarify ambiguities by consulting the user and the tool about the database tables.
                                Once clear, draft a plan. Discuss which tables to use, possible joins, and refine with the user.
                                Create an SQL query, but don't execute it. Instead, present and reflect on it conversationally with the user.
                                Make sure you only query using tables that exist and make sure table name is accurate.
                                {input} and {agent_scratchpad}"""

        PROMPT = PromptTemplate(
            input_variables=["input", "agent_scratchpad"], template=_DEFAULT_TEMPLATE
        )

        tools = [ Tool(
            name="TableSchemaIndex",
            func=lambda q: str(self.schemaindex.as_query_engine().query(q)),
            description="useful for getting tables' scehema information to create a plausible SQL query that could provide the right information to the user",
            return_direct=True,
            ),
        ]
        memory = ConversationBufferMemory(memory_key="chat_history")
        llm = ChatOpenAI(temperature=0, openai_api_key=os.getenv('OPENAI_API_KEY'), model_name='gpt-3.5-turbo-16k') #ChatOpenAI(temperature=0.2, model="gpt-3.5-turbo-16k")

        
        # db_chain = SQLDatabaseChain(llm=llm, database=self.sqldb, prompt=PROMPT, verbose=True)
        temp_db = SQLDatabase.from_uri("sqlite:///dev/chinook.db")
        toolkit = SQLDatabaseToolkit(db=temp_db, llm=llm)
        agent_executor = create_sql_agent(
            toolkit=toolkit, llm=llm, agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, memory=memory, \
            verbose=True,
        )

        # agent_executor.agent.llm_chain.prompt.template = _DEFAULT_TEMPLATE
        return agent_executor





    
    

        