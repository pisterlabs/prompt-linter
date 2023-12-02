# Langchain wraps the Milvus client and provides a few convenience methods for working with documents. 
# It can split documents into chunks, embed them, and store them in Milvus.

import os
import argparse

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Milvus


text_field = "otext"

def main(question, host, port, collection_name):
    embeddings = OpenAIEmbeddings()

    vector_db = Milvus(
        embeddings,
        {"host": host, "port": port},
        collection_name,
        text_field,
    )

    docs = vector_db.similarity_search(question)
     
    print(docs)


# python similar_seatch.py --question your_question --collection_name my_collection
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Similar search for Question Answering over Documents application.")
    parser.add_argument('--question', type=str, required=True, help='Question to search for.')
    parser.add_argument('--host', type=str, default="127.0.0.1", help='Host address for the Milvus server.')
    parser.add_argument('--port', type=str, default="19530", help='Port for the Milvus server.')
    parser.add_argument('--collection_name', type=str, required=True, help='Name of the collection to index the documents into.')

    args = parser.parse_args()

    main(args.question, args.host, args.port, args.collection_name)