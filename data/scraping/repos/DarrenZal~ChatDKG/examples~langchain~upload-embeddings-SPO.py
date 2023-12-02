import os
import pandas as pd
from langchain.vectorstores import Milvus
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import DataFrameLoader
from dotenv import load_dotenv

load_dotenv()

def upload_to_milvus(tsv_file, text_column, collection_name, embedding_model, milvus_connection_args):
    try:
        df = pd.read_csv(tsv_file, sep="\t")
        loader = DataFrameLoader(df, text_column)
        docs = loader.load()

        # Create an instance of HuggingFaceEmbeddings
        # You do not need to use this object to generate embeddings if the Milvus add_documents does that internally
        embedding_function = HuggingFaceEmbeddings(model_name=embedding_model)

        # Create an instance of Milvus and pass the connection arguments
        vector_db = Milvus(
            collection_name=collection_name,
            embedding_function=embedding_function,  # This is used internally by vector_db to generate embeddings
            connection_args=milvus_connection_args,
        )

        # Pass docs directly as the vector_db will handle embedding generation
        vector_db.add_documents(docs)  # Assuming docs is a list of text strings
    except FileNotFoundError:
        print(f"File {tsv_file} not found. Please make sure the file exists.")
    except KeyError as e:
        print(f"Column {e} not found in the file {tsv_file}. Please make sure the column name is correct.")

# Define the embedding model name
embedding_model = "multi-qa-MiniLM-L6-cos-v1"

# Define Milvus connection arguments
milvus_connection_args = {
    "uri": os.getenv("MILVUS_URI"),
    "token": os.getenv("MILVUS_TOKEN"),
    "secure": True,
}

# You would call upload_to_milvus for your TSV file with the correct column name:
upload_to_milvus(
    tsv_file="entities.tsv",
    text_column="Embedding",  # Replace with the actual column name for entities
    collection_name="EntityCollection",
    embedding_model=embedding_model,
    milvus_connection_args=milvus_connection_args
)

# Upload relations without duplicates
upload_to_milvus(
    tsv_file="relations.tsv",
    text_column="Embedding",  # Column containing text for relations
    collection_name="RelationCollection",
    embedding_model=embedding_model,
    milvus_connection_args=milvus_connection_args
)