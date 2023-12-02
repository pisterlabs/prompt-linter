import psycopg
from psycopg.rows import dict_row
import cohere

DATABASE_URL="postgresql://..."

def get_conn():
    conn = psycopg.connect(DATABASE_URL, row_factory=dict_row)
    return conn

def get_cohere_client():
    return cohere.Client("...")


