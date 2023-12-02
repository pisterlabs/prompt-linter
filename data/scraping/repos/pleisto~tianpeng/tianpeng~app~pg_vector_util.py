import psycopg2
from langchain.vectorstores.pgvector import PGVector
import os


def get_conn_string():
    return PGVector.connection_string_from_db_params(
        driver=os.environ.get("PGVECTOR_DRIVER", "psycopg2"),
        host=os.environ.get("PGVECTOR_HOST", "localhost"),
        port=int(os.environ.get("PGVECTOR_PORT", "5432")),
        database=os.environ.get("PGVECTOR_DATABASE", "default_database"),
        user=os.environ.get("PGVECTOR_USER", "admin"),
        password=os.environ.get("PGVECTOR_PASSWORD", "admin123"),
    )


def pg_conn():
    return psycopg2.connect(
        dbname="default_database",  # Replace with your database name
        user="admin",  # Replace with your username
        password="admin123",  # Replace with your password
        host="localhost",  # Replace with your host
        port="5432",  # Replace with your port
    )


def create_extension(cur):
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")


def create_pdf_table(cur):
    cur.execute(
        """
      CREATE TABLE IF NOT EXISTS pdf_table (
          id SERIAL PRIMARY KEY,
          name VARCHAR(100),
          hash VARCHAR(100)
      )
  """
    )


def get_pdf_hash(cur, name):
    cur.execute("SELECT hash FROM pdf_table WHERE name = %s", (name,))
    return cur.fetchone()[0]


def test_init_curs(cur, conn):
    # Create a table
    cur.execute(
        """
      CREATE TABLE IF NOT EXISTS example_table (
          id SERIAL PRIMARY KEY,
          name VARCHAR(100),
          age INTEGER
      )
  """
    )
    conn.commit()
    print("Table created successfully")

    # Insert a record into the table
    cur.execute("INSERT INTO example_table (name, age) VALUES (%s, %s)", ("John", 30))
    conn.commit()
    print("Record inserted successfully")


def init():
    # Establish connection to the PostgreSQL database
    conn = pg_conn()

    # Create a cursor object to interact with the database
    cur = conn.cursor()

    # create extension
    create_extension(cur)

    create_pdf_table(cur)

    # Close cursor and connection
    cur.close()
    conn.close()
