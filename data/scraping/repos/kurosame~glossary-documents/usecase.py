from langchain.schema import Document
from langchain.vectorstores import SupabaseVectorStore

from src.domain.model.documents.documents_repository import DocumentsRepository
from src.middleware.di import di


def from_documents_with_query(
    docs: list[Document], query_name: str
) -> SupabaseVectorStore:
    r = di().get(DocumentsRepository)
    return r.from_with_query(docs, query_name)


def delete_all_documents() -> None:
    r = di().get(DocumentsRepository)
    r.delete_all()
