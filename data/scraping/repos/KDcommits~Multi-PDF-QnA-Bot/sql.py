import os
import pandas as pd
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.agents import create_sql_agent
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_toolkits import SQLDatabaseToolkit

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_KEY')

class SQLQuery:
    def __init__(self):
        self.DB_USERNAME = os.getenv('DB_USERNAME')
        self.DB_PASSWORD = os.getenv('DB_PASSWORD')
        self.DB_HOST = os.getenv('DB_HOST')
        self.DB_NAME = os.getenv('DB_NAME')

    def createDBConnectionString(self):
        db_user = self.DB_USERNAME
        db_Password  = self.DB_PASSWORD
        db_host = self.DB_HOST
        db_name = self.DB_NAME
        connectionString = f"mysql+pymysql://{db_user}:{db_Password}@{db_host}/{db_name}"
        return connectionString
    
    def getSQLSchema(self):
            sql_query = f"""  
            SELECT C.TABLE_NAME, C.COLUMN_NAME, C.DATA_TYPE, T.TABLE_TYPE, T.TABLE_SCHEMA  
            FROM INFORMATION_SCHEMA.COLUMNS C  
            JOIN INFORMATION_SCHEMA.TABLES T ON C.TABLE_NAME = T.TABLE_NAME AND C.TABLE_SCHEMA = T.TABLE_SCHEMA  
            WHERE T.TABLE_SCHEMA = '{self.DB_NAME}' 
            """ 
            mysql_connection_string = self.createDBConnectionString()
            result = pd.read_sql_query(sql_query, mysql_connection_string)
            df = result.infer_objects()
            output=[]
            current_table = ''  
            columns = []  
            for index, row in df.iterrows():
                table_name = f"{row['TABLE_SCHEMA']}.{row['TABLE_NAME']}"  
                column_name = row['COLUMN_NAME']  
                data_type = row['DATA_TYPE']  
                if " " in table_name:
                    table_name= f"[{table_name}]" 
                column_name = row['COLUMN_NAME']  
                if " " in column_name:
                    column_name= f"[{column_name}]" 
                    # If the table name has changed, output the previous table's information  
                if current_table != table_name and current_table != '':  
                    output.append(f"table: {current_table}, columns: {', '.join(columns)}")  
                    columns = []  
                
                # Add the current column information to the list of columns for the current table  
                columns.append(f"{column_name} {data_type}")  
                
                # Update the current table name  
                current_table = table_name  

            # Output the last table's information  
            output.append(f"table: {current_table}, columns: {', '.join(columns)}")
            output = "\n ".join(output)

            return output   

    def createAgentExecutor(self, openAI_model_name="gpt-3.5-turbo"):
        
        mysql_connection_string = self.createDBConnectionString()
        llm = ChatOpenAI(model_name= openAI_model_name )
        db = SQLDatabase.from_uri(mysql_connection_string)
        toolkit = SQLDatabaseToolkit(db=db, llm =llm)
        agent_executor = create_sql_agent(
                                        llm=llm,
                                        toolkit=toolkit,
                                        verbose=True,
                                        return_intermediate_steps=False)
        return agent_executor

    def fetchQueryResult(self,question):
        db_agent = self.createAgentExecutor()
        schema_info = self.getSQLSchema()
        prompt = f'''You are a professional SQL Data Analyst whose job is to fetch results from the SQL database.\
            The SQL Table schema is as follows {schema_info}.\
            The question will be asked in # delimiter. If you are not able to find the answer write "Found Nothing" in response.\
            Do not write anything out of context or on your own.\
            Question : # {question} #'''
        db_agent.return_intermediate_steps=True
        agent_response = db_agent(prompt)
        output = agent_response['output']
        # query = agent_response['intermediate_steps'][-1][0].log.split('\n')[-1].split('Action Input:')[-1].strip().strip('"')
        return output 


# question = "Artificial intelligence in budget ?"
# question = "Find the email id of the salesRepEmployeeNumber for customer number 124"
# sql_obj = SQLQuery()
# response = sql_obj.fetchQueryResult(question )
# print(response)
