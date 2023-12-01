from flask import Blueprint, request, jsonify
from langchain.llms import OpenAI
from langchain.agents import load_tools, initialize_agent
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

embeddings_blueprint = Blueprint('embeddings', __name__)

@embeddings_blueprint.route('/text-to-vector-db', methods=['POST'])
def text_to_vector_db_():
    documents = request.json['documents']
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(documents)
    vector_db = {'vectors': X.toarray().tolist(), 'vectorizer': vectorizer}
    return jsonify(vector_db)
