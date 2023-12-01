import sqlvalidator
import os
import openai
import sqlite3

SQL_path = '''sql_DB/example-covid-vaccinations.sqlite3'''
openai.api_key = "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"
invalidSQL = '''DELETE FROM Customers WHERE CustomerName='Alfreds Futterkiste';'''
validSQL = '''SELECT id, name, surname, standard, drop, city, FROM studentdetails WHERE name = "John" ORDER BY standard ASC'''


def validateSQL(response):
    if not sqlvalidator.parse(response).is_valid():
        print("SQL Invalid")
        return

    badWords = ['INSERT', 'UPDATE', 'DELETE', 'ALTER', 'DROP', 'TRUNCATE',
                'MERGE', 'REPLACE', 'SET', 'COMMIT', 'ROLLBACK', 'SAVEPOINT', 'GRANT',
                'REVOKE', 'DENY', 'EXEC', 'EXECUTE', 'CALL', 'USE']

    return 'Valid'


def getResponse(message,SQLschema):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system",
             "content": f"You assess the users message and respond with only valid SQL according to this schema: {SQLschema}."},
            {"role": "user", "content": f"{message}"}
        ]
    )

    print(completion.choices[0].message['content'])
    print(validateSQL(completion.choices[0].message['content']))


def connect_db(path):
    conn = sqlite3.connect(path)
    # Create a cursor object
    cursor = conn.cursor()

    # Query to retrieve the schema information for all tables
    query = "SELECT sql FROM sqlite_master WHERE type='table';"

    # Execute the query
    cursor.execute(query)

    # Fetch all the results and concatenate them into a single string
    schema_string = ""
    for row in cursor.fetchall():
        schema_string += row[0] + '\n'

    # Close the cursor and the database connection
    cursor.close()
    conn.close()
    print(schema_string)
    getResponse(message='''What was the biggest vaccination rate achieved?''', SQLschema=str(schema_string))
    return conn

connect_db(SQL_path)