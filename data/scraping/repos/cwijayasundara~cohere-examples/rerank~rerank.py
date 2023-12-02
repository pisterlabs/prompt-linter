import os
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

import cohere

co = cohere.Client(os.environ['COHERE_API_KEY'])

import weaviate

auth_config = weaviate.auth.AuthApiKey(
    api_key=os.environ['WEAVIATE_API_KEY'])

client = weaviate.Client(
    url=os.environ['WEAVIATE_API_URL'],
    auth_client_secret=auth_config,
    additional_headers={
        "X-Cohere-Api-Key": os.environ['COHERE_API_KEY'],
    }
)

print(client.is_ready())


def dense_retrieval(query,
                    results_lang='en',
                    properties=["text", "title", "url", "views", "lang", "_additional {distance}"],
                    num_results=5):
    nearText = {"concepts": [query]}

    # To filter by language
    where_filter = {
        "path": ["lang"],
        "operator": "Equal",
        "valueString": results_lang
    }
    response = (
        client.query
        .get("Articles", properties)
        .with_near_text(nearText)
        # .with_where(where_filter)
        .with_limit(num_results)
        .do()
    )

    result = response['data']['Get']['Articles']
    return result


def print_result(result):
    """ Print results with colorful formatting """
    for i, item in enumerate(result):
        print(f'item {i}')
        for key in item.keys():
            print(f"{key}:{item.get(key)}")
            print()
        print()


query = "What is the capital of Canada?"

dense_retrieval_results = dense_retrieval(query, client)

print(dense_retrieval_results)

# Add re rank

query_1 = "What is the capital of Canada?"
results = dense_retrieval(query_1,
                          client,
                          properties=["text", "title", "url", "views", "lang", "_additional {distance}"],
                          num_results=3
                          )

for i, result in enumerate(results):
    print(f"i:{i}")
    print(result.get('title'))
    print(result.get('text'))


def rerank_responses(query, responses, num_responses=10):
    reranked_responses = co.rerank(
        model='rerank-english-v2.0',
        query=query,
        documents=responses,
        top_n=num_responses,
    )
    return reranked_responses


texts = [result.get('text') for result in results]
reranked_text = rerank_responses(query_1, texts)

for i, rerank_result in enumerate(reranked_text):
    print(f"i:{i}")
    print(f"{rerank_result}")
    print()

query_2 = "Who is the tallest person in history?"
results = dense_retrieval(query_2,client)
for i, result in enumerate(results):
    print(f"i:{i}")
    print(result.get('title'))
    print(result.get('text'))
    print()

texts = [result.get('text') for result in results]
reranked_text = rerank_responses(query_2, texts)
for i, rerank_result in enumerate(reranked_text):
    print(f"i:{i}")
    print(f"{rerank_result}")
    print()