from shared_utils import initialize_qdrant_client
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Qdrant
from qdrant_client.http import models as rest

# Initialize Qdrant client and set up collection information
client = initialize_qdrant_client()
collection_name = "Products"
collection_vector_column = "description_vector"

embeddings = OpenAIEmbeddings()

# Create Qdrant vector store instance
# Args here are directly connected to the args found in insert_embeddings_into_collection in shared_utils.py
# Might be worthwile to make connected schema for this
qdrant = Qdrant(
    client=client,
    collection_name=collection_name,
    embeddings=embeddings,
    vector_name=collection_vector_column,
    content_payload_key="content"
)

# Function to perform similarity search and print results, k=1 means we only want to return the top result
def perform_similarity_search(query, filter_condition):
    matched_product = qdrant.similarity_search_with_score(query, k=1, filter=filter_condition)[0]

    result = {
        "query": query,
        "matched_persona": matched_product[0].page_content,
        "metadata": matched_product[0].metadata,
        "score": matched_product[1]
    }

    return result

# Filter condition for brand filtering
# Rich-type support filtering examples at https://qdrant.tech/documentation/concepts/filtering/
specific_brands = ["RVCA", "Cotopaxi"]
brand_filter = rest.Filter(must=[rest.FieldCondition(key="metadata.product_brand", match=rest.MatchAny(any=specific_brands))])

# Perform similarity search for various queries
queries = [
    "I like disco",
    "I like waterfalls",
    "I like hard drugs",
    "I like pondering the meaning of life",
    "I like hoola hooping"
]

for query in queries:
    query_result = perform_similarity_search(query, brand_filter)

    assert query == query_result["query"]

    print(f"Query: {query}")
    print(f"Matched persona: {query_result['matched_persona']}")
    print(f"Metadata: {query_result['metadata']}")
    print(f"Score: {query_result['score']}")
    print()
