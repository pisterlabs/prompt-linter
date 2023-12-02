import pymysql
import environ

from langchain.sql_database import SQLDatabase

env = environ.Env()
environ.Env.read_env()


def connect_to_database():
    try:
        connection = pymysql.connect(
            host=env("DB_HOST"),
            user=env("DB_USER"),
            password=env("DB_PASSWORD"),
            db=env("DB_SCHEMA"),
        )
        return connection
    except Exception as e:
        print(f"Error connecting to the database: {e}")
        return None


def get_database_tables():
    connection = connect_to_database()
    if connection:
        try:
            cursor = connection.cursor()
            cursor.execute("SHOW TABLES")
            return [table[0] for table in cursor.fetchall()]
        except Exception as e:
            print(f"Error fetching database tables: {e}")
    return []


def get_matching_tables(intended_tables):
    all_tables = get_database_tables()
    return [table for table in all_tables if table in intended_tables]


def get_error_tables(intended_tables):
    all_tables = get_database_tables()
    return [table for table in intended_tables if table not in all_tables]


def setup_database():
    db = SQLDatabase.from_uri(
        f"mysql+pymysql://{env('DB_USER')}:{env('DB_PASSWORD')}@{env('DB_HOST')}/{env('DB_SCHEMA')}"
    )
    return db
