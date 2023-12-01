'''Connect to the database as normal postgres database and as vectorstore'''
import os
import psycopg
from langchain.vectorstores.pgvector import PGVector,DistanceStrategy
from langchain.embeddings.openai import OpenAIEmbeddings

class DB():
    def __init__(self) -> None:
        embedding = OpenAIEmbeddings()
        dbname=os.environ.get("POSTGRES_DATABASE", "postgres")
        host=os.environ.get("POSTGRES_HOST", "127.0.0.1")
        port=os.environ.get("POSTGRES_PORT", "5432")
        user=os.environ.get("POSTGRES_USER", "postgres")
        password=os.environ.get("POSTGRES_PASSWORD", "postgres")

        connection_string = 'dbname={dbname} host={host} port=5432 user={user} password={password}'
        connection_string = connection_string.format(
            dbname=dbname,
            host=host,
            port=port,
            user=user,
            password=password
        )

        CONNECTION_STRING = PGVector.connection_string_from_db_params(
    	    driver="psycopg2",
    	    host=host,
    	    port=port,
    	    database=dbname,
    	    user=user,
    	    password=password,
	    )

        self.vectorstore = PGVector(
		    connection_string=CONNECTION_STRING,
		    embedding_function=embedding,
		    distance_strategy=DistanceStrategy.COSINE
	    )

        def __del__(self):
            self.conn.close()
            self.vectorstore.close()



