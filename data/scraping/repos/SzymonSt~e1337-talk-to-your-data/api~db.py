import psycopg2
from simple_ddl_parser import DDLParser
from sqlite_interface import create_connection
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.llms.openai import OpenAI

# Initialize the language model
llm = OpenAI(temperature=0)

# Initialize the database
db = SQLDatabase.from_uri("sqlite:///C://Users//Defozo//Downloads//2//sqlite-dll-win64-x64-3430100//test.d")  # TODO: Replace with your database path

# Create the toolkit
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

# Create the agent
agent = create_sql_agent(llm=llm, toolkit=toolkit, verbose=True)

CREATE_TABLE_CONST=["CREATE", "TABLE"]
DROP_TABLE_CONST=["DROP", "TABLE"]
ALTER_TABLE_CONST=["ALTER", "TABLE"]
CREATE_INDEX_CONST=["CREATE","INDEX"]
DROP_INDEX_CONST=["DROP"," INDEX"]

INIT_TABLES_DISCOVERY_QUERY = """
    SELECT name FROM sqlite_master WHERE type='table';
    """
INIT_TABLE_SCHEMA_DISCOVERY_QUERY = """
    PRAGMA table_info('{}');
    """
INIT_INDICES_DISCOVERY_QUERY = """
    PRAGMA index_list('{}');
    """
INIT_INDEX_INFO_QUERY = """
    PRAGMA index_info('{}');
    """
    
DB_FILE = r"C:\Users\Defozo\Downloads\2\sqlite-dll-win64-x64-3430100\test.d"

class DBConn():
    def __init__(self):
        self.conn = create_connection(DB_FILE)
        self.table_objects = self.initial_tables_discovery()
        self.index_objects = self.initial_indices_discovery()
    
    def execute_query(self, query: str) -> (list,tuple,bool):
        cursor = self.conn.cursor()
        try:
            cursor.execute(query)
        except Exception as e:
            print(e)
            self.conn.rollback()
            cursor.close()
            return None, None, False
        try:
            result = cursor.fetchall()
        except:
            result = None
        self.conn.commit()
        columns = cursor.description
        cursor.close()
        return result, columns, True
    
    def parseIfDDL(self,q: str):
        isDDL = False
        if CREATE_TABLE_CONST[0] in q and CREATE_TABLE_CONST[1] in q:
            print("CREATE TABLE")
            isDDL = True
        if ALTER_TABLE_CONST[0] in q and ALTER_TABLE_CONST[1] in q:
            print("ALTER TABLE")
            isDDL = True
        if DROP_TABLE_CONST[0] in q and DROP_TABLE_CONST[1] in q[q.index(DROP_TABLE_CONST[0]):]:
            print("DROP TABLE")
            isDDL = True
        if CREATE_INDEX_CONST[0] in q and CREATE_INDEX_CONST[1] in q:
            print("CREATE INDEX")
            isDDL = True
        if DROP_INDEX_CONST[0] in q and DROP_INDEX_CONST[1] in q[q.index(DROP_INDEX_CONST[0]):]:
            print("DROP INDEX")
            isDDL = True

        if isDDL:
            self.table_objects = self.initial_tables_discovery()
            self.index_objects = self.initial_indices_discovery()
            return {"tables": self.table_objects, "indices": self.index_objects}
        
        return None

    def initial_tables_discovery(self):
        tmp_tables = {}
        cursor = self.conn.cursor()
        cursor.execute(INIT_TABLES_DISCOVERY_QUERY)
        try:
            tables_results = cursor.fetchall()
        except:
            tables_results = None
        if tables_results is not None:
            for table in tables_results:
                cursor.execute(INIT_TABLE_SCHEMA_DISCOVERY_QUERY.format(table[0]))
                try:
                    schema_results = cursor.fetchall()
                except:
                    schema_results = None
                if schema_results is not None:
                    tmp_tables[table[0]] = {}
                    for column in schema_results:
                        tmp_tables[table[0]][column[0]] = {
                            "type": column[1]
                        }
        cursor.close()
        return tmp_tables

    def initial_indices_discovery(self):
        tmp_indices = {}
        cursor = self.conn.cursor()
        cursor.execute(INIT_TABLES_DISCOVERY_QUERY)
        try:
            tables_results = cursor.fetchall()
        except:
            tables_results = None
        if tables_results is not None:
            for table in tables_results:
                cursor.execute(INIT_INDICES_DISCOVERY_QUERY.format(table[0]))
                try:
                    indices_results = cursor.fetchall()
                except:
                    indices_results = None
                if indices_results is not None:
                    for index in indices_results:
                        tmp_indices[index[0]] = {
                            "table": index[1],
                            "columns": []
                        }
        cursor.close()
        return tmp_indices

def query_splitter(q):
    p = q.replace("(","").replace(")","").replace("\n"," ").replace(";","").replace("\r","").replace(",","")
    p = p.split(" ")
    return p