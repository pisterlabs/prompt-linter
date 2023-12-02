import uuid
from typing import List

from fastapi import APIRouter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.azuresearch import AzureSearch

from api.schemas import azure_cognitive_search as azure_cognitive_search_schemas
from api.settings import azure_cognitive_search as azure_cognitive_search_settings

settings = azure_cognitive_search_settings.AzureCognitiveSearchSettings()
router = APIRouter()
vector_store = None


def get_vector_store() -> AzureSearch:
    if vector_store is not None:
        return vector_store

    embeddings: OpenAIEmbeddings = OpenAIEmbeddings(
        model=settings.openai_model_name,
        deployment=settings.openai_deployment_id,
        openai_api_version=settings.openai_api_version,
        openai_api_base=settings.openai_api_base,
        openai_api_type=settings.openai_api_type,
        openai_api_key=settings.openai_api_key,
        chunk_size=1,
    )
    return AzureSearch(
        azure_search_endpoint=settings.azure_cognitive_search_endpoint,
        azure_search_key=settings.azure_cognitive_search_key,
        index_name=settings.azure_cognitive_search_index_name,
        embedding_function=embeddings.embed_query,
    )


@router.post(
    "/azure_cognitive_search/search",
    response_model=azure_cognitive_search_schemas.SearchResponse,
)
async def search_index(
    request: azure_cognitive_search_schemas.SearchRequest,
):
    vector_store = get_vector_store()
    docs = vector_store.similarity_search(
        query=request.query,
        k=3,
        search_type="hybrid",
    )

    documents: List[azure_cognitive_search_schemas.Document] = []
    for index in range(min(len(docs), request.n_results)):
        documents.append(
            azure_cognitive_search_schemas.Document(
                metadatas=azure_cognitive_search_schemas.Document.Metadatas(
                    source="",
                    title="",
                    score=0.0,
                ),
                text=docs[index].page_content,
            )
        )
    return azure_cognitive_search_schemas.SearchResponse(
        id=str(uuid.uuid4()),
        documents=documents,
    )
