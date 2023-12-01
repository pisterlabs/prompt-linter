from langchain.embeddings import OpenAIEmbeddings

from app.config.milvus_db import get_milvus_client, MILVUS_COLLECTION
from app.util.structured_text_util import determine_extraction_function_based_on_missing_data, \
    update_missing_json_values_with_llm


def parse_text_command(text: str, route: str):
    embedded_text = OpenAIEmbeddings().embed_query(text)
    response = get_milvus_client().search(
        collection_name=MILVUS_COLLECTION,
        data=[embedded_text],
        limit=1,
        output_fields=["text", "route", "start_time", "name", "end_time", "page", "listRows", "label", "operation"]
    )

    entity = response[0][0].get('entity', {})

    function_description = determine_extraction_function_based_on_missing_data(entity)

    if function_description is None:
        if entity.get('route') == route:
            entity['route'] = None
        return entity

    updated_entity = update_missing_json_values_with_llm(
        json_data=entity,
        question=text,
        function_descriptions=function_description
    )

    if updated_entity.get('route') == route:
        updated_entity['route'] = None
    return updated_entity


