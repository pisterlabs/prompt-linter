from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import pandas as pd
import json
import re
import json
import os

class SqlCodeParser:
    """
    This class contains the functions for parsing SQL code.
    """

    CACHE_FILE_NAME = './results/parsed_code_cache.csv'

    def __init__(self, source_directory, source_file_glob_pattern="**/*.sql", use_cache=True, debug=True, cache_file_name=CACHE_FILE_NAME):
        """
        Initialise the class with the source directory and the glob pattern for the SQL files to parse.
        """
        self.source_directory = source_directory
        self.source_file_glob_pattern = source_file_glob_pattern
        self.use_cache = use_cache
        self.debug = debug
        self.cache_file_name = cache_file_name


    def _find_ddl_statements_in_code_segment(self, sql_code):
        """
        Find all the Data Definition Language (DDL) statements in the SQL CODE fragment
        provided and extract the statement type and the name of the database object 
        being created, altered or dropped.

        Output Format:
        An array, containing the name of the database object and the DDL statement type in UPPERCASE text.
        
        Example:
        [{"db_object_name": "EmployeeID", "sql_operation": "CREATE INDEX"}]
        """

        chat = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, verbose=True)
        system_message_prompt = SystemMessagePromptTemplate.from_template("""
            Your are a SQL code parser.

            Find all the Data Definition Language (DDL) statements in the SQL CODE provided and extract the statement type and the name of the database object being created, altered or dropped.

            ## OUTPUT FORMAT ##
            json object array, containing the name of the database object and the DDL statement type in UPPERCASE text.
            Example:
            [{{ "db_object_name": "EmployeeID", "sql_operation": "CREATE INDEX"}}]

            If there are no items, then return an empty json array.
        """)
        example1_prompt = HumanMessagePromptTemplate.from_template('CREATE TABLE "Products"')
        example1_response = AIMessagePromptTemplate.from_template('[{{ "db_object_name": "Products", "sql_operation": "CREATE TABLE"}}]')

        example2_prompt = HumanMessagePromptTemplate.from_template('CONSTRAINT "FK_Products_Categories" FOREIGN KEY')
        example2_response = AIMessagePromptTemplate.from_template('[{{ "db_object_name": "FK_Products_Categories", "sql_operation": "CREATE CONSTRAINT"}}]')

        example3_prompt = HumanMessagePromptTemplate.from_template('create procedure "Sales by Year"')
        example3_response = AIMessagePromptTemplate.from_template('[{{ "db_object_name": "Sales by Year", "sql_operation": "CREATE PROCEDURE"}}]')

        example4_prompt = HumanMessagePromptTemplate.from_template("""
        if exists (select * from sysobjects where id = object_id('dbo.Employee Sales by Country') and sysstat & 0xf = 4)
            drop procedure "dbo"."Employee Sales by Country"
        GO
        """)
        example4_response = AIMessagePromptTemplate.from_template('[{{ "db_object_name": "Employee Sales by Country", "sql_operation": "DROP PROCEDURE"}}]')

        example5_prompt = HumanMessagePromptTemplate.from_template("""
        if exists (select * from sysobjects where id = object_id('dbo.Category Sales for 1997') and sysstat & 0xf = 2)
            drop view "dbo"."Category Sales for 1997"
        GO
        """)
        example5_response = AIMessagePromptTemplate.from_template('[{{ "db_object_name": "Category Sales for 1997", "sql_operation": "DROP VIEW"}}]')

        final_prompt = HumanMessagePromptTemplate.from_template("""
        ## CODE ##
        {sql_code_fragment}

        Output:
        """)
        chat_prompt = ChatPromptTemplate.from_messages([
            system_message_prompt, 
            example1_prompt, example1_response,
            example2_prompt, example2_response,
            example3_prompt, example3_response,
            example4_prompt, example4_response,
            example5_prompt, example5_response,
            final_prompt
        ])

        # get a chat completion from the formatted messages
        llm_response = chat(chat_prompt.format_prompt(sql_code_fragment=sql_code).to_messages())
        try:
            result = json.loads(llm_response.content)
        except:
            print(f"\nFailed to parse the following response into JSON:\n{llm_response.content}\n\nThis content will be excluded.\n\nThe input SQL code was:\n{sql_code}\n\n")
            result = []

        return result


    def _search_all_sql_files_for_ddl_statements(self):
        """
        Reads the SQL code from the files in the source directory and parse the code to 
        extract the Data Definition Language (DDL) statements.

        The output is a dataframe with the following columns:
        - db_object_name: The name of the database object being created, altered or dropped.
        - sql_operation: The type of DDL statement.
        - sql_code: The SQL code that was parsed.

        The sql_operation can be one of the following:
        - CREATE TABLE
        - CREATE INDEX
        - CREATE PROCEDURE
        - CREATE VIEW
        - CREATE CONSTRAINT
        - ALTER TABLE
        - ALTER INDEX
        - ALTER PROCEDURE
        - ALTER VIEW
        - ALTER CONSTRAINT
        - DROP TABLE
        - DROP INDEX
        - DROP PROCEDURE
        - DROP VIEW
        - DROP CONSTRAINT
        """

        if not os.path.exists(self.source_directory):
            raise Exception(f"Source directory does not exist: {self.source_directory}")

        # Load the code to analyse
        loader = DirectoryLoader(
            self.source_directory, glob=self.source_file_glob_pattern, loader_cls=TextLoader, show_progress=True
        )
        documents = loader.load()

        # Split the code into chunks, ideally where there is a GO statement which indicates the end of a significant code block.
        splitter = RecursiveCharacterTextSplitter(
            separators=["GO\n", "go\n", "\n\n", "\n"], chunk_size=2000, chunk_overlap=0, keep_separator=True
        )
        chunks = splitter.split_documents(documents)

        # In debug mode we only process a few chunks to save time and cost.
        sample_chunks = chunks[0:3] if self.debug else chunks

        print(f"Parsing {len(sample_chunks)} code fragments.")
        ddl_statements_df = pd.DataFrame(columns=['db_object_name', 'sql_operation', 'sql_code'])
        for chunk in sample_chunks:
            print(".", end="") # progress indicator
            content = chunk.page_content
            database_objects = self._find_ddl_statements_in_code_segment(content)
            if len(database_objects) > 0:
                temp_df = pd.DataFrame(database_objects)
                temp_df['sql_code'] = content
                ddl_statements_df = pd.concat([ddl_statements_df, temp_df], ignore_index=True)

        return ddl_statements_df


    def find_ddl_statements(self):
        """
        Finds all DDL statements in the SQL code in the source directory.
        Uses cached results if they exist and the use_cache parameter is set to True.
        
        The output is a dataframe with the following columns:
        - db_object_name: The name of the database object being created, altered or dropped.
        - sql_operation: The type of DDL statement.
        - sql_code: The SQL code that was parsed.
        """
        if self.use_cache and os.path.exists(self.cache_file_name):
            df = pd.read_csv(self.cache_file_name)
        else:
            df = self._search_all_sql_files_for_ddl_statements()
            df.to_csv(self.cache_file_name, index=False)
        
        return df
            

    def extract_procedure_declaration_from_code(self, procedure_name, sql_code):
        """
        Extract the procedure declaration from the SQL code.
        
        Uses a regular expression to fetch the code between the CREATE PROCEDURE 
        statement and the GO statement.
        """
        regex_pattern = r'(CREATE PROCEDURE +?"?%s"?.+?(\nGO|$))' % procedure_name
        # regex_pattern = r'(CREATE PROCEDURE +?"?%s"?[^GO]+?(\nGO|$))' % procedure_name
        match = re.search(regex_pattern, sql_code, re.DOTALL | re.IGNORECASE)
        # match = re.search(regex_pattern, sql_code, re.IGNORECASE)
        if match:
            procedure_code = match.group(1)
        else:
            procedure_code = None
        return procedure_code


    def find_tables_manipulated_by_procedure(self, procedure_name, sql_code):
        """
        Find all the database tables that are manipulated by the procedure.

        Returns a list of dictionaries containing the name of the table and the 
        DML statement type (i.e. SELECT, INSERT, UPDATE, or DELETE).

        Example:
        [
            { "table_name": "Order Details", "sql_operation": "SELECT"},
            { "table_name": "Order Details", "sql_operation": "INSERT"},
            { "table_name": "Order Details", "sql_operation": "UPDATE"},
            { "table_name": "Order Details", "sql_operation": "DELETE"},
        ]
        """
        chat = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, verbose=True)
        system_message_prompt = SystemMessagePromptTemplate.from_template("""
            Your are a SQL code parser.

            Find all the database tables that are manipulated by the {procedure_name} stored procedure.
            Find all the tables that are queried, inserted into, updated or deleted from and extract the 
            table name and database operation type (i.e. SELECT, INSERT, UPDATE, or DELETE).

            ## OUTPUT FORMAT ##
            json object array, containing the name of the table and the DML statement type in UPPERCASE text.
            Example:
            [{{ "table_name": "Order Details", "sql_operation": "SELECT"}}]

            If there are no items, then return an empty json array.
        """)
        example1_prompt = HumanMessagePromptTemplate.from_template("""
        CREATE PROCEDURE CustOrdersDetail @OrderID int
        AS
        SELECT ProductName,
            UnitPrice=ROUND(Od.UnitPrice, 2),
            Quantity,
            Discount=CONVERT(int, Discount * 100), 
            ExtendedPrice=ROUND(CONVERT(money, Quantity * (1 - Discount) * Od.UnitPrice), 2)
        FROM Products P, [Order Details] Od
        WHERE Od.ProductID = P.ProductID and Od.OrderID = @OrderID
        go
        """)
        example1_response = AIMessagePromptTemplate.from_template('[{{ "table_name": "Products", "sql_operation": "SELECT"}}, {{ "table_name": "Order Details", "sql_operation": "SELECT"}}]')

        example2_prompt = HumanMessagePromptTemplate.from_template("""
        CREATE PROCEDURE CustOrdersOrders @CustomerID nchar(5)
        AS
        SELECT OrderID, 
            OrderDate,
            RequiredDate,
            ShippedDate
        FROM Orders
        WHERE CustomerID = @CustomerID
        ORDER BY OrderID
        GO
        """)
        example2_response = AIMessagePromptTemplate.from_template('[{{ "table_name": "Orders", "sql_operation": "SELECT"}}]')

        final_prompt = HumanMessagePromptTemplate.from_template("{sql_code_fragment}")

        chat_prompt = ChatPromptTemplate.from_messages([
            system_message_prompt, 
            example1_prompt, example1_response,
            example2_prompt, example2_response,
            final_prompt
        ])

        # get a chat completion from the formatted messages
        llm_response = chat(chat_prompt.format_prompt(procedure_name=procedure_name, sql_code_fragment=sql_code).to_messages())
        result = json.loads(llm_response.content)
        return result

