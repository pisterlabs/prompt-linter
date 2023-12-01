import time

from flask import Blueprint, Flask, jsonify, make_response, request
from flask_cors import CORS, cross_origin
from dotenv import load_dotenv
import os
import openai
from .utils.embeddings import select_embeddings
from .utils.autocomplete import autocomplete_query_suggestions, generate_queries_for_schema
from .utils.indexes import select_index
from .utils.repair import repair_query_suggestions
from .utils import connect_to_db, explain_query
from threading import Thread
import logging

logger = logging.getLogger(__name__)

api_bp = Blueprint('api_bp', __name__)
app = Flask(__name__)
CORS(app, resources={
     r"/*": {"origins": "http://localhost:3000", "supports_credentials": True}})
app.config['CORS_HEADERS'] = 'Content-Type'

load_dotenv()
openai.api_key = os.environ.get('OPENAI_API_KEY')
TOTAL_AUTOCOMPLETE_TIME = os.environ.get('TOTAL_AUTOCOMPLETE_TIME', 3)
TOTAL_REPAIR_TIME = os.environ.get('TOTAL_REPAIR_TIME', 10)


@api_bp.route('/discover', methods=['OPTIONS', 'POST'])
@cross_origin(origin='http://localhost:3000', supports_credentials=True)
def discover():
    if request.method == 'OPTIONS':
        return make_response(jsonify({}), 200)

    db, error = connect_to_db(request)
    dialect = request.json.get('dialect', 'postgresql')
    if error:
        return error

    index_engine = select_index()

    embeddings = select_embeddings()
    docsearch = index_engine.read_index(db, embeddings)

    texts = []
    metadatas = []
    table_queries = {}
    schema_dict = request.json.get('schema', None)
    tables = schema_dict.keys()
    table_info_dict = request.json.get('tables_info', None)
    for table in tables:
        info = table_info_dict[table].replace('\r', ' ').replace('\t', ' ').replace('\n', ' ') if table_info_dict else None
        texts.append(info)
        metadatas.append({'type': 'schema'})
        if os.environ.get('GET_SAMPLE_QUERIES', False):
            table_queries[table] = info

    # add to a vector search using embeddings
    docsearch = index_engine.read_index_contents(texts, embeddings, metadatas)
    index_engine.write_index(db, docsearch)

    # add sample queries
    if os.environ.get('GET_SAMPLE_QUERIES', False):
        # run the generate_and_save_sample_queries in a new thread to not block the code execution
        thread = Thread(target=generate_and_save_sample_queries,
                        args=(table_queries, dialect, schema_dict, docsearch))
        thread.start()

    index_engine.write_index(db, docsearch)

    response = jsonify({"status": 'OK'})
    return response


def generate_and_save_sample_queries(table_queries, dialect, schema_dict, docsearch):
    logger.info('Started generating sample queries')
    for info in table_queries.values():
        queries = generate_queries_for_schema(info, schema_dict, dialect)
        if (queries and len(queries) > 0):
            for query in queries:
                docsearch.add_texts([query], [{'type': 'query'}])
    logger.info('Finished generating sample queries')


@api_bp.route('/autocomplete', methods=['OPTIONS', 'POST'])
@cross_origin(origin='http://localhost:3000', supports_credentials=True)
def autocomplete():
    if request.method == 'OPTIONS':
        return make_response(jsonify({}), 200)

    db, error = connect_to_db(request)
    dialect = request.json.get('dialect', 'postgresql')
    if error:
        return error

    index_engine = select_index()

    embeddings = select_embeddings()
    docsearch = index_engine.read_index(db, embeddings)
    if docsearch is not None:
        query = request.json.get('query', None)
        if query:
            # execute query autocompletion
            result = autocomplete_query_suggestions(
                query.strip(), docsearch, dialect)
            response = jsonify({'suggestions': result})
            return response
        else:
            return make_response(jsonify({'error': 'No query provided'}), 400)
    else:
        return make_response(jsonify({'error': 'Error retrieving index'}), 500)


@api_bp.route('/repair', methods=['OPTIONS', 'POST'])
@cross_origin(origin='http://localhost:3000', supports_credentials=True)
def repair():
    if request.method == 'OPTIONS':
        return make_response(jsonify({}), 200)

    # st = time.time()
    db, error = connect_to_db(request)
    dialect = request.json.get('dialect', 'postgresql')
    if error:
        return error

    index_engine = select_index()

    embeddings = select_embeddings()
    docsearch = index_engine.read_index(db, embeddings)
    if docsearch is not None:
        query = request.json.get('query', None)
        error_message = request.json.get('error_message', None)
        if query and error_message:
            # execute query repair
            result = repair_query_suggestions(
                query.strip(), error_message.strip(), docsearch, dialect)
            response = jsonify({'suggestions': result})
            return response
        else:
            return make_response(jsonify({'error': 'No query or error provided'}), 400)
    else:
        return make_response(jsonify({'error': 'Error retrieving index'}), 500)


@api_bp.route('/add', methods=['OPTIONS', 'POST'])
@cross_origin(origin='http://localhost:3000', supports_credentials=True)
def add():
    if request.method == 'OPTIONS':
        return make_response(jsonify({}), 200)

    db, error = connect_to_db(request)
    if error:
        return error

    index_engine = select_index()
    embeddings = select_embeddings()
    docsearch = index_engine.read_index(db, embeddings)
    if docsearch is not None:
        query = request.json.get('query', None)
        if query:
            docsearch.add_texts([query], [{'type': 'query'}])
            index_engine.write_index(db, docsearch)
            response = jsonify({"status": 'OK'})
            return response
        else:
            return make_response(jsonify({'error': 'No query provided'}), 400)
    else:
        return make_response(jsonify({'error': 'Cannot query without docsearch base'}), 500)
