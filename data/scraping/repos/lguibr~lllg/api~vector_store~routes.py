from flask import Blueprint, request, jsonify, current_app as app
from api.context.routes import get_context_by_name
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import DirectoryLoader
from services.firebase import bucket, db
import os

vectorstore = Blueprint("vectorstore", __name__)
VECTORSTORE_FOLDER = "temp/vectorstores"

SOURCE_FOLDER = "temp/sources"
CHROMA_DB_FOLDER = "temp/chroma_dbs"
EMBEDDING_FUNCTION = OpenAIEmbeddings()


@vectorstore.route("/api/vectorstore", methods=["POST"])
def create_vectorstore():
    logger = app.logger
    try:
        data = request.get_json()
        vectorstore_name = data["name"]
        description = data["description"]
        context_names = data["topics"]
        vector_store_dir = os.path.join(CHROMA_DB_FOLDER, vectorstore_name)

        download_contexts_to_source_dir(context_names, SOURCE_FOLDER)
        documents = process_documents(SOURCE_FOLDER)
        persist_chroma_documents(documents, vector_store_dir)
        logger.info(f"vector_store_dir: {vector_store_dir}")
        upload_dir_to_bucket(vector_store_dir, bucket)

        save_to_firestore(
            db, vectorstore_name, description, vector_store_dir, context_names
        )

        return jsonify({"message": "Vectorstore created successfully."}), 201
    except Exception as e:
        return jsonify({"message": str(e)}), 404


@vectorstore.route("/api/vectorstore/<id>", methods=["GET"])
def get_vectorstore(id):
    ref = db.collection("vectorstores").document(id)
    vectorstore = ref.get()
    if vectorstore.exists:
        return jsonify(vectorstore.to_dict()), 200
    else:
        return jsonify({"message": f"Vectorstore {id} not found."}), 404


@vectorstore.route("/api/vectorstore", methods=["GET"])
def list_vectorstores():
    refs = db.collection("vectorstores").stream()
    vectorstores = [{doc.id: doc.to_dict()} for doc in refs]
    return jsonify({"vectorstores": vectorstores}), 200


@vectorstore.route("/api/vectorstore/<id>", methods=["DELETE"])
def delete_vectorstore(id):
    try:
        ref = db.collection("vectorstores").document(id)
        if ref.get().exists:
            ref.delete()

        blob_name = f"vectorstore/{id}"
        blob = bucket.blob(blob_name)
        if blob.exists():
            blob.delete()

        return jsonify({"message": "Vectorstore deleted successfully."}), 200
    except Exception as e:
        return jsonify({"message": str(e)}), 404


@vectorstore.route("/api/vectorstore/<id>/query", methods=["POST"])
def question_answer(id):
    logger = app.logger

    try:
        local_folder = os.path.join(CHROMA_DB_FOLDER, id)
        blob_folder = f"{CHROMA_DB_FOLDER}/{id}"
        download_dir_from_bucket(bucket, blob_folder, local_folder)
    except Exception as e:
        return jsonify({"message": str(e)}), 404
    try:
        vectorstore = Chroma(
            embedding_function=OpenAIEmbeddings(),
            persist_directory=local_folder,
        )

        data = request.get_json()
        query = data["query"]
        chain = load_qa_with_sources_chain(
            llm=ChatOpenAI(model_name="gpt-4", temperature=0),
        )

        docs = vectorstore.similarity_search(query, k=6)
        logger.info(f"Found {len(docs)} documents")
        response = chain({"input_documents": docs, "question": query})

        return jsonify({"data": response["output_text"]}), 200
    except Exception as e:
        return jsonify({"message": str(e)}), 404


def download_dir_from_bucket(bucket, source_folder, destination_folder):
    blobs = bucket.list_blobs(prefix=source_folder)
    for blob in blobs:
        filename = blob.name.replace(source_folder, destination_folder)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        blob.download_to_filename(filename)


def download_contexts_to_source_dir(context_names, source_dir):
    for context_name in context_names:
        context = get_context_by_name(context_name)
        if context:
            files = context["files"]
            for file in files:
                download_dir_from_bucket(bucket, file, os.path.join(source_dir, file))


def process_documents(source_dir):
    loader = DirectoryLoader(source_dir, show_progress=True)
    unsplitted_documents = loader.load()
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=0
    )
    return text_splitter.split_documents(documents=unsplitted_documents)


def persist_chroma_documents(documents, chroma_db_dir):
    chroma = Chroma.from_documents(
        documents=documents,
        embedding=EMBEDDING_FUNCTION,
        persist_directory=chroma_db_dir,
    )
    chroma.persist()
    chroma = None


def upload_dir_to_bucket(source_dir, bucket):
    for dirpath, dirnames, filenames in os.walk(source_dir):
        for file in filenames:
            local_file = os.path.join(dirpath, file)
            blob_path = os.path.relpath(local_file, source_dir)
            blob = bucket.blob(os.path.join(source_dir, blob_path))
            blob.upload_from_filename(local_file)


def save_to_firestore(
    db, vectorstore_name, description, vector_store_dir, context_names
):
    ref = db.collection("vectorstores").document(vectorstore_name)
    ref.set(
        {
            "name": vectorstore_name,
            "description": description,
            "blob_name": vector_store_dir,
            "contexts": context_names,
        }
    )
