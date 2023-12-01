import psycopg2
import openai
import datachain_backend.openai_loader as openai_loader
import csv

import csv
import psycopg2
import re

class SQLChain:
    """
    A class that represents a SQLChain object for connecting to a PostgreSQL database and performing SQL operations.
    """

    def __init__(self):
        """
        Initializes a SQLChain object with default database connection parameters.
        """
        self.database = "datachain-test"
        self.host = "localhost"
        self.user = "postgres"
        self.password = "password"
        self.port = "5432"
        self.conn = None
    
    def connect(self):
        """
        Connects to the PostgreSQL database using the specified connection parameters.
        """
        try:
            print("Trying to Connect...")
            self.conn = psycopg2.connect(
                database=self.database, 
                user=self.user, 
                password=self.password, 
                host=self.host,
                port=self.port)
            print("Connected to database")
        except psycopg2.Error as e:
            print(f"Unable to connect to the database: {e}")

    def create_table(self, table_name, csv):
        """
        Creates a table in the database based on the provided table name and CSV data.

        Args:
            table_name (str): The name of the table to be created.
            csv (str): The CSV data used to determine the table structure.

        Returns:
            str: The SQL query for creating the table.
        """
        #Table name to lower cases and remove spaces and special characters make sure it starts with a letter
        table_name = table_name.lower().replace(" ", "_").replace("-", "_").replace(".", "_").replace(",", "_").replace(";", "_").replace(":", "_").replace("!", "_").replace("?", "_").replace("(", "_").replace(")", "_").replace("[", "_").replace("]", "_").replace("{", "_").replace("}", "_").replace("'", "_").replace('"', "_").replace("/", "_").replace("\\", "_").replace("|", "_").replace("=", "_").replace("+", "_").replace("*", "_").replace("&", "_").replace("^", "_").replace("%", "_").replace("$", "_").replace("#", "_").replace("@", "_")
        # Create a query with openai
        prompt = f"""Based on the first two lines of csv below, write a query to create a table. 
        Makse sure to Name the table {table_name}. 
        Note that in SQL, column names cannot start with a number. 
        Make sure there is no character limit for the data. 
        Make sure that no columns start with a number, rename if necessary:
        [CSV Data]
        {csv}
        [END CSV Data]
        """

        messages = [] 
        messages.append({"role": "user", "content": prompt})
        chat_response = openai_loader.chat_completion_request(messages)
        query = chat_response.json()["choices"][0]["message"]['content']
        return query

    def get_table_schema(self, table_name):
        """
        Gets the schema of the specified table in the PostgreSQL database.
        Args:
            table_name (str): The name of the table to get the schema of.
        Returns:
            list: A list of tuples representing the table schema.
        """
        cursor = self.conn.cursor()
        cursor.execute(f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{table_name}';")
        rows = cursor.fetchall()
        return rows
    
    def create_query(self, prompt, table_name, model = None):
        """
        Creates SQL query based on natural language prompt and table name.

        Args:
            prompt (str): The natural language prompt.
            table_name (str): The name of the table to be created.
        
        Returns:
            str: The SQL query for creating the table.
        """
        #Check table's schema
        table_schema = self.get_table_schema(table_name)
        #Create prompt for llm
        prompt = f"""
            You are given a table named {table_name} with the following schema:
            {table_schema}
            Write an SQL query to answer the prompt: {prompt}
            BONT WRAP IN CODEBLOCK, WRAP  YOUR QUERY IN [START QUERY] AND [END QUERY]
            SQL Query:
        """
        messages = [] 
        messages.append({"role": "user", "content": prompt})
        if model:
            chat_response = openai_loader.chat_completion_request(messages, model=model)
        else:
            chat_response = openai_loader.chat_completion_request(messages)
        print(chat_response.json())
        #ADd error handling in case of API errors
        query = chat_response.json()["choices"][0]["message"]['content']
        match = re.search(r'\[START QUERY\]\n(.*?)\n\[END QUERY\]', query, re.DOTALL)

        if match:
            query = match.group(1)
        return query
    
    def execute_query(self, query):
        """
        Executes the provided SQL query on the connected PostgreSQL database.

        Args:
            query (str): The SQL query to be executed.
        """
        cursor = self.conn.cursor()
        try:
            print(f"Executing Query...: {query}")
            cursor.execute(query)
            self.conn.commit()
            print("Query executed successfully")
        except psycopg2.Error as e:
            print(f"Unable to execute query: {e}")
            self.conn.rollback()


    def disconnect(self):
        """
        Disconnects from the PostgreSQL database.
        """
        try:
            print("Trying to Disconnect...")
            self.conn.close()
            print("Disconnected from database")
        except psycopg2.Error as e:
            print(f"Unable to disconnect from the database: {e}")


    def check_table_exists(self, table_name):
        """
        Checks if a table with the specified name exists in the PostgreSQL database.

        Args:
            table_name (str): The name of the table to check.

        Returns:
            bool: True if the table exists, False otherwise.
        """
        with self.conn.cursor() as cursor:
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE  table_name   = %s
                );
            """, (table_name,))
            return cursor.fetchone()[0]
        
    def delete_table(self, table_name):
        """
        Deletes the table with the specified name from the PostgreSQL database.

        Args:
            table_name (str): The name of the table to be deleted.
        """
        with self.conn.cursor() as cursor:
            cursor.execute(f"DROP TABLE {table_name};")
            self.conn.commit()

        
    def insert_data_from_csv(self, file_object, table_name):
        """
        Inserts data from a CSV file into the specified table in the PostgreSQL database.

        Args:
            file_object (file): The file object containing the CSV data.
            table_name (str): The name of the table to insert the data into.
        """
        reader = csv.reader(file_object)
        next(reader)  # Skip the header row
        for row in reader:
            row = [None if value == 'NA' else value for value in row]  # Replace 'NA' with None
            with self.conn.cursor() as cursor:
                placeholders = ', '.join(['%s'] * len(row))
                query = f"INSERT INTO {table_name} VALUES ({placeholders});"
                cursor.execute(query, row)
        self.conn.commit()
        print("Data inserted successfully")    

    def fetch_data(self, table_name, limit=5):
        """
        Fetches data from the specified table in the PostgreSQL database.

        Args:
            table_name (str): The name of the table to fetch data from.
            limit (int, optional): The maximum number of rows to fetch. Defaults to 5.

        Returns:
            list: A list of tuples representing the fetched rows.
        """
        with self.conn.cursor() as cursor:
            cursor.execute(f"SELECT * FROM {table_name} LIMIT {limit};")
            rows = cursor.fetchall()
            for row in rows:
                print(row)

        return rows

    # Add more methods as needed
