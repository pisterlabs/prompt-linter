import os
from flask import Blueprint, request, jsonify, current_app as app
from services.firebase import db, bucket

from langchain.retrievers.merger_retriever import MergerRetriever

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI


retriever = Blueprint("retriever", __name__)

RETRIEVER_FOLDER = "temp/retriever"


@retriever.route("/api/retriever", methods=["POST"])
def create_retriever():
    try:
        data = request.get_json()
        ref = db.collection("retrievers").document(data["name"])
        ref.set(
            {
                "name": data["name"],
                "description": data["description"],
                "retrievers": data["retrievers"],
            }
        )
        return jsonify({"message": "Retriever created successfully."}), 201
    except Exception as e:
        return jsonify({"message": str(e)}), 404


@retriever.route("/api/retriever/<id>", methods=["GET"])
def get_retriever(id):
    ref = db.collection("retrievers").document(id)
    retriever = ref.get()
    if retriever.exists:
        return jsonify(retriever.to_dict()), 200
    else:
        return jsonify({"message": f"Retriever {id} not found."}), 404


@retriever.route("/api/retriever", methods=["GET"])
def list_retrievers():
    refs = db.collection("retrievers").stream()
    retrievers = [{doc.id: doc.to_dict()} for doc in refs]
    return jsonify({"retrievers": retrievers}), 200


@retriever.route("/api/retriever/<id>", methods=["DELETE"])
def delete_retriever(id):
    try:
        ref = db.collection("retrievers").document(id)
        if ref.get().exists:
            ref.delete()
        return jsonify({"message": "Retriever deleted successfully."}), 200
    except Exception as e:
        return jsonify({"message": str(e)}), 404


@retriever.route("/api/retriever/<id>", methods=["PUT"])
def update_retriever(id):
    try:
        data = request.get_json()
        ref = db.collection("retrievers").document(id)
        if ref.get().exists:
            ref.update(
                {
                    "name": data["name"],
                    "description": data["description"],
                    "retrievers": data["retrievers"],
                }
            )
        return jsonify({"message": "Retriever updated successfully."}), 200
    except Exception as e:
        return jsonify({"message": str(e)}), 404


@retriever.route("/api/retriever/<id>/load", methods=["GET"])
def load_retriever(id):
    try:
        # get retriever from firestore
        ref = db.collection("retrievers").document(id)
        retriever = ref.get().to_dict()

        if retriever:
            vectorstores = retriever["retrievers"]
            load_vectorstores(vectorstores)
            return jsonify({"message": f"Retriever {id} loaded successfully."}), 200
        else:
            return jsonify({"message": f"Retriever {id} not found."}), 404
    except Exception as e:
        return jsonify({"message": str(e)}), 404


def load_vectorstores(vectorstores):
    # for each vectorstore in retriever, download it from the bucket
    for vectorstore in vectorstores:
        local_folder = os.path.join(RETRIEVER_FOLDER, vectorstore)
        blob_folder = f"temp/chroma_dbs/{vectorstore}"
        download_dir_from_bucket(bucket, blob_folder, local_folder)


def create_chroma_instance(directory):
    """
    Create a Chroma instance with OpenAIEmbeddings
    """
    embeddings = OpenAIEmbeddings()
    chroma = Chroma(
        persist_directory=directory,
        embedding_function=embeddings,
    )
    return chroma


def build_retriever(retriever_id):
    app.logger.info(f"Building retriever {retriever_id}")
    try:
        # get retriever from firestore
        ref = db.collection("retrievers").document(retriever_id)
        retriever = ref.get().to_dict()
        app.logger.info("retriever")
        app.logger.info(retriever)

        if retriever:
            vectorstores = retriever["retrievers"]
            app.logger.info("vectorstores")
            app.logger.info(vectorstores)
            retrievers = []
            for vectorstore in vectorstores:
                # Get the vectorstore data from firestore
                vectorstore_ref = db.collection("vectorstores").document(vectorstore)
                vectorstore_data = vectorstore_ref.get().to_dict()

                # Get the local directory for this vectorstore
                local_folder = os.path.join(
                    RETRIEVER_FOLDER, vectorstore_data["blob_name"]
                )

                if os.path.exists(local_folder):
                    retriever = create_chroma_instance(
                        directory=local_folder
                    ).as_retriever()
                    retrievers.append(retriever)
                else:
                    app.logger.error(f"Vectorstore directory {local_folder} not found.")

            # Create MergerRetriever
            if retrievers:
                lotr = MergerRetriever(retrievers=retrievers)
                return lotr
            else:
                app.logger.error(f"No retrievers created for {retriever_id}.")
                return None

        else:
            app.logger.error(f"Retriever {retriever_id} not found.")
            return None

    except Exception as e:
        app.logger.error(f"Failed to build retriever: {str(e)}")
        return None


@retriever.route("/api/retriever/<id>", methods=["POST"])
def question_retriever(id):
    data = request.get_json()
    query = data["query"]
    app.logger.info(f"query : {query}")

    retriever = build_retriever(id)
    app.logger.info("retriever")
    app.logger.info(retriever)
    qa = ConversationalRetrievalChain.from_llm(
        ChatOpenAI(model_name="gpt-4", temperature=0),
        retriever,
    )
    result = qa({"question": query})
    app.logger.info(f"Retriever result: {result}")
    return result


def download_dir_from_bucket(bucket, source_folder, destination_folder):
    blobs = bucket.list_blobs(prefix=source_folder)
    for blob in blobs:
        filename = blob.name.replace(source_folder, destination_folder)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        blob.download_to_filename(filename)
