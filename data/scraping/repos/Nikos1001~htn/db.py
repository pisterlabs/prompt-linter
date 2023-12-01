from psycopg2 import OperationalError
import psycopg2
import cohere
import json

connector = psycopg2.connect('postgresql://hiatus:zK8yCsqmKmIdEKSn0_8WYA@pet-indri-3361.g95.cockroachlabs.cloud:26257/defaultdb?sslmode=verify-full')
connector.autocommit = True

def execute_query(query):
    cursor = connector.cursor()
    try:
        cursor.execute(query)
        print("Query success")
    except OperationalError as err:
        print(f"Error {err}")

# execute_query('DROP TABLE deck_list')
execute_query("CREATE TABLE IF NOT EXISTS deck_list (id SERIAL PRIMARY KEY, deck JSON)")

def add_to_db(deck):
    deck_str = json.dumps(deck).replace('\'', '')
    print(deck_str)
    add_query = f"INSERT INTO deck_list (deck) VALUES ('{deck_str}')"
    execute_query(add_query)

def retrieve_db():
    deck_list = []

    cursor = connector.cursor()
    try:
        cursor.execute("SELECT * FROM deck_list")
        for deck in cursor.fetchall():
            deck_list.append(deck[1])
    except OperationalError as err:
        print(f"The error '{err}' occurred")

    return deck_list

