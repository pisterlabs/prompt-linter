"""
## Analyze Customer Feedback using the Cohere and OpenSearch Airflow providers

This DAG ingests customer feedback data from a mock API into OpenSearch, uses 
OpenSearch to query the data for relevant feedback, Cohere to create vector embeddings of
the feedback, as well as sentiment analysis and then loads the embeddings and sentiment
into OpenSearch. Finally, it performs a KNN search on the embeddings to find similar
feedback to a given search term.
"""

from airflow.decorators import dag, task
from airflow.models.baseoperator import chain
from pendulum import datetime
from airflow.operators.empty import EmptyOperator
from airflow.providers.cohere.operators.embedding import CohereEmbeddingOperator
from airflow.providers.cohere.hooks.cohere import CohereHook
from airflow.providers.opensearch.operators.opensearch import (
    OpenSearchAddDocumentOperator,
    OpenSearchCreateIndexOperator,
    OpenSearchQueryOperator,
)
from airflow.providers.opensearch.hooks.opensearch import OpenSearchHook
import uuid
import requests
from include.classification_examples import SENTIMENT_EXAMPLES

# control API input
NUM_CUSTOMERS = 2

# this is the search term for which we want to find similar feedback
TESTIMONIAL_SEARCH_TERM = "Using this product for MLOps and loving it!"

# subset query parameters if you make changes here, make sure to also adjust
# the mock API in include/mock_api/app.py
CUSTOMER_LOCATION = "Switzerland"
PRODUCT_TYPE = "cloud service A"
AB_TEST_GROUP = "A"
FEEDBACK_SEARCH_TERMS = "UI OR UX OR user interface OR user experience"
MAX_NUMBER_OF_RESULTS = 1000

# this is the length of the embeddings returned by Cohere
MODEL_VECTOR_LENGTH = 768

# connection ids
COHERE_CONN_ID = "cohere_default"
OPENSEARCH_CONN_ID = "opensearch_default"
OPENSEARCH_INDEX_NAME = "customer_feedback"


@dag(
    start_date=datetime(2023, 10, 18),
    schedule=None,
    catchup=False,
)
def analzye_customer_feedback():
    # --------------------------------------------- #
    # Ingest customer feedback data into OpenSearch #
    # --------------------------------------------- #

    @task.branch
    def check_if_index_exists(index_name: str, conn_id: str) -> str:
        "Check if the index already exists in OpenSearch."
        client = OpenSearchHook(open_search_conn_id=conn_id, log_query=True).client
        is_index_exist = client.indices.exists(index_name)
        if is_index_exist:
            return "index_exists"
        return "create_index"

    create_index = OpenSearchCreateIndexOperator(
        task_id="create_index",
        opensearch_conn_id=OPENSEARCH_CONN_ID,
        index_name=OPENSEARCH_INDEX_NAME,
        index_body={
            "settings": {
                "index": {
                    "number_of_shards": 1,
                    "knn": True,
                    "knn.algo_param.ef_search": 100,
                }
            },
            "mappings": {
                "properties": {
                    "customer_feedback": {"type": "text"},
                    "customer_rating": {"type": "integer"},
                    "customer_id": {"type": "keyword"},
                    "timestamp": {"type": "date"},
                    "customer_location": {"type": "keyword"},
                    "product_type": {"type": "keyword"},
                    "ab_test_group": {"type": "keyword"},
                    "embeddings": {
                        "type": "knn_vector",
                        "dimension": MODEL_VECTOR_LENGTH,
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",
                            "engine": "nmslib",
                        },
                    },
                    "sentiment_prediction": {"type": "keyword"},
                    "sentiment_confidence": {"type": "float"},
                }
            },
        },
    )

    index_exists = EmptyOperator(task_id="index_exists")

    @task
    def get_customer_feedback(num_customers: int) -> list:
        "Query the mock API for customer feedback data."
        r = requests.get(
            f"http://customer_ticket_api:5000/api/data?num_reviews={num_customers}"
        )
        return r.json()

    all_customer_feedback = get_customer_feedback(num_customers=NUM_CUSTOMERS)

    @task
    def customer_feedback_to_dict_list(customer_feedback: list):
        "Convert the customer feedback data into a list of dictionaries."
        list_of_feedback = []
        for customer in customer_feedback:
            unique_line_id = uuid.uuid5(
                name=" ".join(
                    [str(customer["customer_id"]), str(customer["timestamp"])]
                ),
                namespace=uuid.NAMESPACE_DNS,
            )
            kwargs = {"doc_id": str(unique_line_id), "document": customer}

            list_of_feedback.append(kwargs)

        return list_of_feedback

    list_of_document_kwargs = customer_feedback_to_dict_list(
        customer_feedback=all_customer_feedback
    )

    add_lines_as_documents = OpenSearchAddDocumentOperator.partial(
        task_id="add_lines_as_documents",
        opensearch_conn_id=OPENSEARCH_CONN_ID,
        trigger_rule="none_failed",
        index_name=OPENSEARCH_INDEX_NAME,
    ).expand_kwargs(list_of_document_kwargs)

    # -------------------------------------------- #
    # Query customer feedback data from OpenSearch #
    # -------------------------------------------- #

    search_for_relevant_feedback = OpenSearchQueryOperator(
        task_id="search_for_relevant_feedback",
        opensearch_conn_id=OPENSEARCH_CONN_ID,
        index_name=OPENSEARCH_INDEX_NAME,
        query={
            "size": MAX_NUMBER_OF_RESULTS,
            "query": {
                "bool": {
                    "must": [
                        {
                            "match": {
                                "customer_feedback": {
                                    "query": FEEDBACK_SEARCH_TERMS,
                                    "analyzer": "english",
                                    "fuzziness": "AUTO",
                                }
                            }
                        }
                    ],
                    "filter": [
                        {"term": {"customer_location": CUSTOMER_LOCATION}},
                        {"term": {"ab_test_group": AB_TEST_GROUP}},
                        {"term": {"product_type": PRODUCT_TYPE}},
                    ],
                },
            },
        },
    )

    @task
    def reformat_relevant_reviews(search_results: dict) -> list:
        "Reformat the relevant reviews from the OpenSearch query results."
        ids = [x["_id"] for x in search_results["hits"]["hits"]]
        reviews_of_interest = [x["_source"] for x in search_results["hits"]["hits"]]
        reviews_with_id = []
        for id, review in zip(ids, reviews_of_interest):
            review["id"] = id
            reviews_with_id.append(review)
        return reviews_of_interest

    relevant_reviews = reformat_relevant_reviews(
        search_results=search_for_relevant_feedback.output
    )

    @task
    def get_feedback_texts(review_of_interest: dict) -> str:
        "Get the feedback text from the relevant reviews."
        feedback_text = review_of_interest["customer_feedback"]
        return feedback_text

    feedback_texts = get_feedback_texts.expand(review_of_interest=relevant_reviews)

    # --------------------------------------- #
    # Perform sentiment analysis              #
    # on relevant customer feedback           #
    # and get embeddings using the Cohere API #                             #
    # --------------------------------------- #

    @task
    def get_sentiment(input_text: str, sentiment_examples: list, conn_id: str) -> float:
        "Get the sentiment of the customer feedback using the Cohere API."
        co = CohereHook(conn_id=conn_id).get_conn

        response = co.classify(
            model="large",
            inputs=[input_text],
            examples=sentiment_examples,
        )

        print(input_text)
        print(response.classifications)

        return {
            "prediction": response.classifications[0].prediction,
            "confidence": response.classifications[0].confidence,
        }

    sentiment_scores = get_sentiment.partial(
        conn_id=COHERE_CONN_ID, sentiment_examples=SENTIMENT_EXAMPLES
    ).expand(input_text=feedback_texts)

    get_embeddings = CohereEmbeddingOperator.partial(
        task_id="get_embeddings",
        conn_id=COHERE_CONN_ID,
    ).expand(input_text=feedback_texts)

    @task
    def combine_reviews_embeddings_and_sentiments(
        reviews: list, embeddings: list, sentiments: list
    ) -> list:
        "Combine the reviews, embeddings and sentiments into a single list of dictionaries."
        review_with_embeddings = []
        for review, embedding, sentiment in zip(reviews, embeddings, sentiments):
            review_with_embeddings.append(
                {
                    "review": review,
                    "embedding": embedding[0],
                    "sentiment_prediction": sentiment["prediction"],
                    "sentiment_confidence": sentiment["confidence"],
                }
            )
        return review_with_embeddings

    full_data = combine_reviews_embeddings_and_sentiments(
        reviews=relevant_reviews,
        embeddings=get_embeddings.output,
        sentiments=sentiment_scores,
    )

    @task
    def load_embeddings_into_opensearch(full_data: dict, conn_id: str) -> None:
        "Load the embeddings and sentiment into OpenSearch."
        client = OpenSearchHook(open_search_conn_id=conn_id, log_query=True).client
        client.update(
            index=OPENSEARCH_INDEX_NAME,
            id=full_data["review"]["id"],
            body={
                "doc": {
                    "embeddings": [float(x) for x in full_data["embedding"]],
                    "sentiment_prediction": full_data["sentiment_prediction"],
                    "sentiment_confidence": full_data["sentiment_confidence"],
                }
            },
        )

    load_embeddings_obj = load_embeddings_into_opensearch.partial(
        conn_id=OPENSEARCH_CONN_ID
    ).expand(full_data=full_data)

    # ------------------------------------------------------- #
    # Query OpenSearch for the most similar testimonial using #
    # KNN on the embeddings and filter for positive sentiment #
    # ------------------------------------------------------- #

    get_embeddings_testimonial_search_term = CohereEmbeddingOperator(
        task_id="get_embeddings_testimonial_search_term",
        conn_id=COHERE_CONN_ID,
        input_text=TESTIMONIAL_SEARCH_TERM,
    )

    @task
    def prep_search_term_embeddings_for_query(embeddings: list) -> list:
        "Prepare the embeddings for the OpenSearch query."
        return [float(x) for x in embeddings[0]]

    search_term_embeddings = prep_search_term_embeddings_for_query(
        embeddings=get_embeddings_testimonial_search_term.output
    )

    search_for_testimonial_candidates = OpenSearchQueryOperator(
        task_id="search_for_testimonial_candidates",
        opensearch_conn_id=OPENSEARCH_CONN_ID,
        index_name=OPENSEARCH_INDEX_NAME,
        query={
            "size": 10,
            "query": {
                "bool": {
                    "must": [
                        {
                            "knn": {
                                "embeddings": {
                                    "vector": search_term_embeddings,
                                    "k": 10,
                                }
                            }
                        }
                    ],
                    "filter": [
                        {"term": {"sentiment_prediction": "positive"}},
                    ],
                }
            },
        },
    )

    @task
    def print_testimonial_candidates(search_results: dict) -> None:
        "Print the testimonial candidates from the OpenSearch query results."
        for result in search_results["hits"]["hits"]:
            print("Customer ID: ", result["_source"]["customer_id"])
            print("Customer feedback: ", result["_source"]["customer_feedback"])
            print("Customer location: ", result["_source"]["customer_location"])
            print("Customer rating: ", result["_source"]["customer_rating"])
            print("Customer sentiment: ", result["_source"]["sentiment_prediction"])
            print(
                "Customer sentiment confidence: ",
                result["_source"]["sentiment_confidence"],
            )

    chain(
        check_if_index_exists(
            index_name=OPENSEARCH_INDEX_NAME, conn_id=OPENSEARCH_CONN_ID
        ),
        [create_index, index_exists],
        add_lines_as_documents,
        search_for_relevant_feedback,
        relevant_reviews,
        feedback_texts,
        load_embeddings_obj,
        get_embeddings_testimonial_search_term,
        search_for_testimonial_candidates,
        print_testimonial_candidates(
            search_results=search_for_testimonial_candidates.output
        ),
    )


analzye_customer_feedback()
