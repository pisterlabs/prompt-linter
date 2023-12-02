from models.models import Document, DocumentChunk, DocumentChunkMetadata
from chunk_text import chunk_text
from db.datastore import QdrantManager
from openai_manager.base import OpenAiManager
from uuid import uuid4

openai_manager = OpenAiManager()
db_manager = QdrantManager()


def upsert_document(document: Document):
    document_content = document.text
    chunks = chunk_text(document_content)

    embeddings = openai_manager.get_embeddings(chunks)

    document_chunks = []

    for chunk, embedding in zip(chunks, embeddings):
        document_chunk = DocumentChunk(
            id=uuid4().hex,
            text=chunk,
            metadata=DocumentChunkMetadata(),
            embedding=embedding,
        )

        document_chunks.append(document_chunk)

    ids = [chunk.id for chunk in document_chunks]
    payloads = [{"chunk": chunk.text} for chunk in document_chunks]
    embeddings = [chunk.embedding for chunk in document_chunks]

    response = db_manager.upsert_points(ids, payloads, embeddings)

    return ids


def delete_datastore():
    status = db_manager.delete_collection()
    return status
