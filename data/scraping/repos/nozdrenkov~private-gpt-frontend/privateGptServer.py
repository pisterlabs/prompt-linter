#!/usr/bin/env python3
"""Flask server to find relevant entries in documents using the power of embeddings."""

from typing import Tuple, Any
import os
import faulthandler
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from constants import CHROMA_SETTINGS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
import threading

faulthandler.enable()

load_dotenv()

_EMBEDDINGS_MODEL_NAME = os.getenv("EMBEDDINGS_MODEL_NAME")
_PERSIST_DIRECTORY = os.getenv('PERSIST_DIRECTORY')
_TARGET_SOURCE_CHUNKS = int(os.getenv('TARGET_SOURCE_CHUNKS', 4))
_DEFAULT_TOP = 5

app = Flask(__name__)
CORS(app)

_embeddings = HuggingFaceEmbeddings(model_name=_EMBEDDINGS_MODEL_NAME)
_db = Chroma(persist_directory=_PERSIST_DIRECTORY,
             embedding_function=_embeddings, client_settings=CHROMA_SETTINGS)

lock = threading.Lock()


@app.route('/query', methods=['POST'])
def handle_query() -> Tuple[Any, int]:
    """Handles the POST request at '/query' endpoint."""
    try:
        data = request.get_json(force=True)
        with lock:
            relevant_docs = _db.similarity_search_with_score(
                query=data.get('query'),
                distance_metric="cos",
                k=int(data.get('top', _DEFAULT_TOP)),
            )

        results = [
            {
                'source': doc[0].metadata['source'],
                'page_content': doc[0].page_content,
                'relevance': doc[1]
            } for doc in relevant_docs
        ] if relevant_docs else []

        return jsonify(results), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=4242)
