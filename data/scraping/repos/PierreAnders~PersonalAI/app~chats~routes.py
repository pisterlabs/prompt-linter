from app.chats import bp
import os
from flask import request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from langchain.vectorstores import Chroma
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from glob import glob
from langchain.document_loaders import DirectoryLoader
from app.chats import bp
from app.chats.service import chat_with_data_service

@bp.route('/chatWithData/<model>', methods=['POST'])
@jwt_required()
def chat_with_data(model):
    data = request.get_json()
    result = chat_with_data_service(model, data)
    return jsonify(result)
