"""
Build and querying the RAG model. 
"""
import openai
import chromadb
from utils import get_embedding_function

def response(query):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"{query}"},
        ]
    )
    return response['choices'][0]['message']['content']


def rag_response(query, context):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Please answer the query using the context provided."},
            {"role": "user", "content": f"query: {query}. context: {context}"},
        ]
    )
    return response['choices'][0]['message']['content']


def get_rag_context(query, client, num_docs=3):
    collection = client.get_collection(name="reuters_collection", embedding_function=get_embedding_function())
    results = collection.query(
        query_texts=[query],
        n_results=num_docs
    )
    contexts = [doc.replace("\n", " ") for doc in results['documents'][0]]
    return contexts


def main():
    client = chromadb.PersistentClient(path="../chromadb/test_db")

    query = "When do farmers sow sugar beet in Holland?"
    contexts = get_rag_context(query, client)
    default_response = response(query)
    ragged_response = rag_response(query, ";".join(contexts))
    print(f"Query: {query}")
    print(f"Default response: {default_response}")
    print(f"RAG response: {ragged_response}")
    print("\n")

    query = "Name a finance minister of West Germany."
    contexts = get_rag_context(query, client)
    default_response = response(query)
    ragged_response = rag_response(query, ";".join(contexts))
    print(f"Query: {query}")
    print(f"Default response: {default_response}")
    print(f"RAG response: {ragged_response}")
    print("\n")

    query = "What was the inflation rate in Indonesia in 1986?"
    contexts = get_rag_context(query, client)
    default_response = response(query)
    ragged_response = rag_response(query, ";".join(contexts))
    print(f"Query: {query}")
    print(f"Default response: {default_response}")
    print(f"RAG response: {ragged_response}")
    print("\n")


if __name__ == "__main__":
    main()