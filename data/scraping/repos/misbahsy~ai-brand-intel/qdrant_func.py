from langchain.docstore.document import Document
from operator import itemgetter


content_payload_key = "page_content"
metadata_payload_key = "metadata"

def test_similarity_search(query,k,filter,embedding_func, collection_name, client,**kwargs):
        results = test_similarity_search_with_score(query, k, filter,embedding_func, collection_name, client)
        return list(map(itemgetter(0), results))

def test_similarity_search_with_score(query, k, filter, embedding_func, collection_name, client):
    embedding = embedding_func(query)
    results = client.search(
        collection_name=collection_name,
        query_vector=embedding,
        query_filter=test_qdrant_filter_from_dict(filter),
        with_payload=True,
        limit=k,
    )
    return [
        (
            test_document_from_scored_point(
                result,content_payload_key, metadata_payload_key
            ),
            result.score,
        )
        for result in results
    ]

def test_document_from_scored_point(
    scored_point,
    content_payload_key,
    metadata_payload_key,
):
    return Document(
        page_content=scored_point.payload.get(content_payload_key),
        metadata=scored_point.payload.get(metadata_payload_key) or {},
    )

def test_qdrant_filter_from_dict(filter):
    if filter is None or 0 == len(filter):
        return None

    from qdrant_client.http import models as rest

    return rest.Filter(
        must=[
            rest.FieldCondition(
                key=f"{metadata_payload_key}.{key}",
                match=rest.MatchValue(value=value),
            )
            for key, value in filter.items()
        ]
    )

def delete_from_client(filter, collection_name, client):
    from qdrant_client.http import models
    if test_qdrant_filter_from_dict(filter) is None:
        deletion_status = client.delete_collection(
        collection_name=collection_name
        )
        return deletion_status
    deletion_status = client.delete(
        collection_name=collection_name,
        points_selector=models.FilterSelector(
            filter=test_qdrant_filter_from_dict(filter),
        ),
    )
    return deletion_status.status.COMPLETED=='completed'