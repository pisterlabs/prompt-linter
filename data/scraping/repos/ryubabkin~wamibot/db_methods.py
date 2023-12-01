import os
import settings as sts
from langchain.document_loaders import Docx2txtLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from utils.readers import get_google_service, google_doc_to_text, google_folder_to_text

def save_document_to_db(
        source: str, # 'file', 'google_folder', 'google_doc'
        db_directory: str,
        item: str = None,
        collection_name: str = 'default',
        chunks_size: int = 500
):
    if source == 'file':
        documents = Docx2txtLoader(item.strip()).load()
    elif source == 'google_folder':
        documents = google_folder_to_text(
            folder_id=item,
            token_path='./credentials/token.json'
        )
    elif source == 'google_doc':
        _, doc_service = get_google_service(token_path='./credentials/token.json')
        documents = [google_doc_to_text(
            file_id=item,
            doc_service=doc_service
        )]
    else:
        raise ValueError("Invalid source")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunks_size,
        chunk_overlap=50
    )
    all_splits = text_splitter.split_documents(documents)
    db = Chroma.from_documents(
        documents=all_splits,
        persist_directory=db_directory,
        collection_name=collection_name,
        embedding=OpenAIEmbeddings()
    )
    db.persist()


def load_data_from_db(
        db_directory: str,
        collection_name: str = 'ALL'
) -> Chroma:
    db = Chroma(
        persist_directory=db_directory,
        collection_name=collection_name,
        embedding_function=OpenAIEmbeddings()
    )
    return db


def delete_collection(
        db_directory: str,
        collection_name: str
):
    db = load_data_from_db(
        db_directory=db_directory,
        collection_name=collection_name
    )
    db.delete_collection()
    db.persist()


def reset_db(
        db_directory: str = sts.DB_FOLDER,
    ):
        vectordb = Chroma(persist_directory=sts.DB_FOLDER, embedding_function=OpenAIEmbeddings())
        for doc in os.listdir('docs'):
            print(doc)
            if doc.split('.') != 'docx':
                continue
            doc_name = doc.split('.')[0]
            delete_collection(
                db_directory=db_directory,
                collection_name=doc_name
            )
            save_document_to_db(
                source='file',
                item=os.path.join('docs', doc),
                db_directory=db_directory,
                collection_name=doc_name,
                chunks_size=500
            )
