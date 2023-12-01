from typing import Dict, List
from asyncpg import Connection
from models.model_manager import ModelManager
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json
from pprint import pprint
import re

whitespace_pattern = re.compile(r"\s+")


def split_document(
    document: str,
    chunk_size=256,
    chunk_overlap=0,
    separators=["\n\n", "\n", " ", ""],
    keep_separator=True,
) -> Dict:
    """
    Split the document into chunks of text small enough to be embedded.
    chunk_size: is measured: by number of characters.
    chunk_overlap: is measured: by number of characters.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=separators,
        keep_separator=keep_separator,
    )
    # split_documents , create_documents
    texts = text_splitter.split_text(document)
    # texts = text_splitter.split_documents(document)
    # texts = [re.sub(r"\s+", " ", text.strip()) for text in texts]
    texts = [whitespace_pattern.sub(" ", text.strip()) for text in texts]

    return texts


async def check_for_duplicates(connection: Connection, text_chunk) -> str:
    """
    Check if a document with the same title, type, page number, doc path,
    and text_chunk already exists in the database and return its UUID.

    Returns:
    - UUID of the duplicate document if it exists, None otherwise
    """
    query = """
        SELECT guid FROM documents 
        WHERE 
            text_chunk = $1
        LIMIT 1
    """
    uuid = await connection.fetchval(query, text_chunk)
    return uuid


async def insert_to_db(
    connection: Connection, document: Dict, embed_model: ModelManager
):
    document_title = document["document_title"]
    type = document["type"]
    page_number = document["page_number"]
    doc_path = document["doc_path"]
    tables = document["tables"]
    images = document["images"]
    metadata = document["metadata"]

    # Splitting the document into smaller text_chunks
    document_chunks = split_document(
        document["text_chunk"],
        chunk_size=256,
        chunk_overlap=64,
        separators=["\n\n", "\n", ".", " "],
        keep_separator=True,
    )
    print(f"*** Number of text_chunks: {len(document_chunks)}")

    for i, text in enumerate(document_chunks):
        print(f"\n*** Text chunk {i}, length: {len(text)}:\n {text}")
    # Embedding the text_chunks as one batch
    try:
        response = await embed_model.embed_documents(document_chunks)
    except Exception as e:
        print(f"*** Exception during embedding: {e}")
        raise e

    document_guid = await check_for_duplicates(connection, document["text_chunk"])
    if document_guid is None:
        try:
            # Start transaction
            async with connection.transaction():
                # Insert document details and return id and guid
                inserted_row = await connection.fetchrow(
                    """
                    INSERT INTO documents(document_title, type, text_chunk, page_number, doc_path, tables, images, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    RETURNING id, guid
                    """,
                    document_title,
                    type,
                    document["text_chunk"],  # the whole document text
                    page_number,
                    doc_path,
                    tables,
                    images,
                    json.dumps(metadata),
                )

                document_id, document_guid = inserted_row["id"], inserted_row["guid"]

                # Insert embeddings
                for embedding_data in response["data"]:
                    await connection.execute(
                        """
                        INSERT INTO embeddings(document_id, embedding_vector)
                        VALUES ($1, $2)
                        """,
                        document_id,
                        embedding_data["embedding"],
                    )
                print(
                    f"*** Text chunk inserted with id: {document_id} and guid: {document_guid}"
                )
        except Exception as e:
            print(f"*** Exception during database transaction: {e}")
            raise e
    else:
        print(f"*** Text chunk already exists with guid: {document_guid}")

    response["uuid"] = str(document_guid)
    return response


async def get_similar_text(
    connection: Connection,
    embed_model: ModelManager,
    text_for_search: str,
    n_top: int,
    search_in_embeddings_only: bool = True,
) -> List[str]:
    # Step 1: Embed the input text
    try:
        embedding = await embed_model.embed_documents([text_for_search])
    except Exception as e:
        print(f"*** Exception during embedding: {e}")
        raise e

    search_embedding = embedding["data"][0]["embedding"]

    if search_in_embeddings_only:
        # Search only in embeddings table
        query = """
            SELECT document_id, embedding_vector <-> $1 AS distance
            FROM embeddings
            ORDER BY distance
            LIMIT $2
        """
        results = await connection.fetch(query, search_embedding, n_top)
        document_ids = [result["document_id"] for result in results]
        print(f"*** Document ids: {document_ids}")
    else:
        # Search in both embeddings and documents tables
        query_embeddings = """
            SELECT document_id, embedding_vector <-> $1 AS distance
            FROM embeddings
            ORDER BY distance
            LIMIT $2
        """
        query_documents = """
            SELECT id AS document_id, embedding_vector <-> $1 AS distance
            FROM documents
            WHERE NOT is_deleted
            ORDER BY distance
            LIMIT $2
        """
        results_embeddings = await connection.fetch(
            query_embeddings, search_embedding, n_top
        )
        results_documents = await connection.fetch(
            query_documents, search_embedding, n_top
        )

        # Combine and sort the results
        combined_results = sorted(
            results_embeddings + results_documents, key=lambda x: x["distance"]
        )[:n_top]
        document_ids = [result["document_id"] for result in combined_results]

    # Retrieve the corresponding text chunks
    query_text_chunks = """
        SELECT text_chunk
        FROM documents
        WHERE id = ANY($1)
    """
    text_chunks = await connection.fetch(query_text_chunks, document_ids)

    return [chunk["text_chunk"] for chunk in text_chunks]


async def vectorize_document(
    file_path: str, connection: Connection, embed_model: ModelManager
):
    with open(file_path, "r") as file:
        document_content = file.read()

    # Splitting the document into text_chunks with the size to provide a good context
    document_chunks = split_document(
        document_content,
        chunk_size=1000,
        chunk_overlap=0,
        separators=["\n\n", "\n"],
        keep_separator=True,
    )
    for chunk in document_chunks:
        # Process each chunk with insert_to_db
        document = {
            "document_title": None,
            "type": 3,
            "page_number": None,
            "doc_path": file_path,
            "tables": [],
            "images": [],
            "metadata": {},
            "text_chunk": chunk,
        }
        await insert_to_db(connection, document, embed_model)
