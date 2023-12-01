import sys

import flask
from flask import Flask
import os
from dotenv import load_dotenv
from langchain import OpenAI

from handlers.compose import compose_handler
from handlers.create import create_handler
from handlers.query import query_handler
from handlers.list import list_handler

from handlers.upload import upload_doc_handler, upload_link_handler
from models.knowledgebase_model import Knowledgebase
from models.statics_model import ResponseStatics, g_index, Models

app = Flask(__name__)
load_dotenv()
OpenAI.openai_api_key = os.getenv('OPENAI_API_KEY')


@app.route('/create', methods=['GET'])
async def create():
    """Create a knowledgebase and return a knowlegebase ID"""
    # Create a unique ID for this creation request
    return ResponseStatics.build_creation_response(create_handler())



@app.route('/index/doc/add', methods=['POST'])
def upload_doc():
    # Extract the knowledgebase_id from the request
    knowledgebase_id = flask.request.form.get('knowledgebase_id')

    # Validate the knowledgebase ID
    if not upload_doc_handler(knowledgebase_id, flask.request.files['file']):
        return ResponseStatics.build_api_error("Invalid knowledgebase ID or none provided")
    return ResponseStatics.build_upload_success_message()


@app.route('/index/link/add', methods=['POST'])
async def upload_link():
    # Extract the knowledgebase_id from the request
    knowledgebase_id, link = flask.request.form.get('knowledgebase_id'), flask.request.form.get('url')

    # Validate the knowledgebase ID
    if not await upload_link_handler(knowledgebase_id, link):
        return ResponseStatics.build_api_error("Invalid knowledgebase ID or none provided")
    return ResponseStatics.build_upload_success_message()


@app.route('/compose', methods=['GET'])
async def compose():
    # Extract the knowledgebase_id from the request
    knowledgebase_id = flask.request.args.get('knowledgebase_id')

    # Validate the knowledgebase ID
    tokens = await compose_handler(knowledgebase_id)
    if not tokens:
        return ResponseStatics.build_api_error("Invalid knowledgebase ID or none provided")

    return ResponseStatics.build_compose_success_message(tokens)


@app.route('/query', methods=['GET'])
async def query():
    # Extract the query from the request (GET)
    query = flask.request.args.get('query')

    nodes = int(flask.request.args.get('nodes')) if flask.request.args.get('nodes') else 3

    # Assert that the number of nodes is valid
    if nodes < 1 or nodes > 10:
        return ResponseStatics.build_api_error("Invalid number of nodes provided [1,10] is valid")

    model = flask.request.args.get('model')

    # Assert that the model is valid and in the list of models
    if model and model not in Models.get_models():
        return ResponseStatics.build_api_error("Invalid model provided")

    # Extract the knowledgebase_id from the request
    knowledgebase_id = flask.request.args.get('knowledgebase_id')

    return await query_handler(knowledgebase_id, query, nodes, model)


@app.route('/index/list', methods=['GET'])
async def list_index():
    return await list_handler()


def init():
    app.run(host="0.0.0.0", port=8182, debug=True)


if __name__ == '__main__':
    sys.exit(init())
