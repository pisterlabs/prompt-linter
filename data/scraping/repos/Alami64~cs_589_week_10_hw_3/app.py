import os
from dotenv import load_dotenv, find_dotenv
import cohere
import weaviate
from utils import dense_retrieval, print_result, keyword_search

# Load environment variables
_ = load_dotenv(find_dotenv())


co = cohere.Client(os.environ['COHERE_API_KEY'])
auth_config = weaviate.auth.AuthApiKey(api_key=os.environ['WEAVIATE_API_KEY'])
client = weaviate.Client(
    url=os.environ['WEAVIATE_API_URL'],
    auth_client_secret=auth_config,
    additional_headers={"X-Cohere-Api-Key": os.environ['COHERE_API_KEY']}
)

def rerank_responses(query, responses, num_responses=10):
    """
    Re-rank the given responses based on the query.
    """
    reranked_responses = co.rerank(
        model='rerank-english-v2.0',
        query=query,
        documents=responses,
        top_n=num_responses
    )
    return reranked_responses

# Example 1: Dense Retrieval
query = "What is the capital of Canada?"
dense_retrieval_results = dense_retrieval(query, client)
print(f"Dense Retrieval Results for query, {query}:")
print_result(dense_retrieval_results)

# Example 2: Keyword Search with ReRank
query_1 = "What is the capital of Canada?"
results = keyword_search(query_1, client, 
                         properties=["text", "title", "url", "views", "lang", "_additional {distance}"], 
                         num_results=500)
texts = [result.get('text') for result in results]
reranked_text = rerank_responses(query_1, texts)
for i, rerank_result in enumerate(reranked_text):
    print(f"Result {i+1} for query '{query_1}':")
    print(f"{rerank_result}\n")

# Example 3: Improving Dense Retrieval with ReRank
query_2 = "Who is the tallest person in history?"
results = dense_retrieval(query_2, client)
texts = [result.get('text') for result in results]
reranked_text = rerank_responses(query_2, texts)
for i, rerank_result in enumerate(reranked_text):
    print(f"Result {i+1} for query '{query_2}':")
    print(f"{rerank_result}\n")