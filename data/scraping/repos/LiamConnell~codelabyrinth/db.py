# db.py
import os
import psycopg2
from langchain.vectorstores.pgvector import PGVector
from psycopg2 import OperationalError
from psycopg2.extras import Json, DictCursor


class Database:
    def __init__(self):
        self.conn = None

    def connect(self):
        if not self.conn:
            try:
                self.conn = psycopg2.connect(
                    host="localhost",
                    port=5432,
                    dbname="postgres",
                    user="postgres",
                    password=os.getenv('POSTGRES_PASSWORD')
                )
            except OperationalError as e:
                print(f"The error '{e}' occurred")

    def close(self):
        if self.conn:
            self.conn.close()

    def execute_query(self, query, params=None):
        self.connect()
        with self.conn.cursor(cursor_factory=DictCursor) as cursor:
            cursor.execute(query, params)
            self.conn.commit()

    def fetch_query(self, query, params=None):
        self.connect()
        with self.conn.cursor(cursor_factory=DictCursor) as cursor:
            cursor.execute(query, params)
            result = cursor.fetchall()
            self.conn.commit()
        return result


def install_extension():
    db = Database()
    db.execute_query("CREATE EXTENSION vector;")
