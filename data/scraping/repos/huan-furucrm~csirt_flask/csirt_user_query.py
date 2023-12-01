# Question about Timescale we want the model to answer
import numpy as np
from llama_index.embeddings import openai
from pgvector.psycopg2 import register_vector
from llama_index import OpenAIEmbedding, LangchainEmbedding

input = "How is Timescale used in IoT?"

def set_user_query(input):
    return input


# Helper function: get embeddings for a text
def get_embeddings(text):
    # Create an instance of OpenAIEmbedding
    openai_embedding = OpenAIEmbedding(
        model="text-embedding-ada-002"
    )
    # Create an instance of LangchainEmbedding with OpenAIEmbedding
    embedding = LangchainEmbedding(openai_embedding, embed_batch_size=1)

    # Use the embedding instance to generate embeddings
    input = text.replace("\n", " ")
    embeddings = embedding.embed(input)

    return embedding


# Helper function: Get top 1 most similar documents from the database
def get_top1_similar_docs(query_embedding, conn):
    print("Ndim is ...")
    print(query_embedding.ndim)
    # embedding_array = np.array(query_embedding)
    # Register pgvector extension
    register_vector(conn)
    cur = conn.cursor()
    # Get the top 1 most similar documents using the KNN <=> operator
    cur.execute("SELECT text FROM data_salesforce_data_index ORDER BY embedding <=> %s LIMIT 1", (query_embedding[0],))
    top1_docs = cur.fetchall()
    return top1_docs

def get_top2_similar_docs(query_embedding, conn):
    print("Ndim is ...")
    print(query_embedding.ndim)
    # embedding_array = np.array(query_embedding)
    # Register pgvector extension
    register_vector(conn)
    cur = conn.cursor()
    # Get the top 1 most similar documents using the KNN <=> operator
    cur.execute("SELECT text FROM data_salesforce_data_index ORDER BY embedding <-> %s LIMIT 2", (query_embedding[0],))
    top2_docs = cur.fetchall()
    return top2_docs


def get_tops_similar_docs(query_embedding, conn, top = 1):
    # Register pgvector extension
    register_vector(conn)
    cur = conn.cursor()
    # Get the tops most similar documents using the KNN <=> operator
    
    cur.execute("SELECT text FROM data_salesforce_data_index ORDER BY embedding <-> %s LIMIT %s", (query_embedding, str(top)))
    top2_docs = cur.fetchall()
    return top2_docs

def get_query_index(conn):
     # Register pgvector extension
    register_vector(conn)
    cur = conn.cursor()
    # Get the tops most similar documents using the KNN <=> operator
    
    cur.execute("SELECT embedding FROM data_salesforce_data_index_query")
    query_index = cur.fetchall()
    return query_index[0]