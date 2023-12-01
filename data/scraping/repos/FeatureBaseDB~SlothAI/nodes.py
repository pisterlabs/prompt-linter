from google.cloud import ndb

from flask import Blueprint, jsonify, request
from flask import current_app as app
import flask_login
from flask_login import current_user

from SlothAI.web.models import Node, Pipeline
from SlothAI.lib.template import Template
from SlothAI.lib.util import merge_extras

node_handler = Blueprint('node_handler', __name__)

# API HANDLERS
@node_handler.route('/nodes/list', methods=['GET'])
@flask_login.login_required
def nodes_list():
    # get the user and their tables
    username = current_user.name
    api_token = current_user.api_token
    dbid = current_user.dbid
    nodes = Node.fetch(uid=current_user.uid)

    return jsonify(nodes)


@node_handler.route('/nodes/<node_id>/detail', methods=['GET'])
@flask_login.login_required
def get_node(node_id):
    # Get the user and their tables
    username = current_user.name
    
    # Fetch the node by node_id
    node = Node.get(uid=current_user.uid, node_id=node_id)

    if node:
        return jsonify(node)
    else:
        return jsonify({"error": "Not found", "message": "The requested node was not found."}), 404


@node_handler.route('/nodes/validate/openai', methods=['POST'])
@flask_login.login_required
def validate_openai():
    uid = current_user.uid

    if request.is_json:
        json_data = request.get_json()

    if json_data.get('openai_token', None):
        import openai
        openai.api_key = json_data.get('openai_token')
        try:
            result = openai.Model.list()
        except:
            return jsonify({'error': "Invalid Token.", "message": "That token did not validate."}), 400
    else:
        return jsonify({"error": "Invalid JSON", "message": "'openai_token' key with data is required in the request JSON."}), 400
    
    return jsonify({"result": "Token validated. Adding new node..."}), 200

# TODO ADD NODE UPDATE
@node_handler.route('/nodes', methods=['POST'])
@node_handler.route('/nodes/create', methods=['POST'])
@flask_login.login_required
def node_create():
    uid = current_user.uid

    if request.is_json:
        json_data = request.get_json()

        if 'node' in json_data and isinstance(json_data['node'], dict) and json_data['node'].get('template_id', '').lower() != 'none':
            node_data = json_data['node']

            template_service = app.config['template_service']
            template = template_service.get_template(template_id=node_data.get('template_id'))

            merged_extras = merge_extras(template.get('extras', {}), node_data.get('extras', {}))
           
            created_node = Node.create(
                name=node_data.get('name'),
                uid=uid,
                extras=merged_extras,
                processor=node_data.get('processor'),
                template_id=node_data.get('template_id')
            )

            if created_node:
                return jsonify(created_node), 201
            else:
                return jsonify({"error": "Creation failed", "message": "Failed to create the node."}), 500
        else:
            return jsonify({"error": "Invalid JSON", "message": "'node' key with dictionary data is required in the request JSON."}), 400
    else:
        return jsonify({"error": "Invalid JSON", "message": "The request body must be valid JSON data."}), 400


@node_handler.route('/nodes/<node_id>', methods=['DELETE'])
@node_handler.route('/nodes/<node_id>/delete', methods=['DELETE'])
@flask_login.login_required
def node_delete(node_id):
    node = Node.get(uid=current_user.uid, node_id=node_id)
    if node:
        # Fetch all pipelines
        pipelines = Pipeline.fetch(uid=current_user.uid)

        # Check if the node is in any pipeline
        is_in_pipeline = any(node_id in pipeline.get('node_ids', []) for pipeline in pipelines)

        if is_in_pipeline:
            return jsonify({"error": "Node is in a pipeline", "message": "This node cannot be deleted until it's removed from the pipelines using it."}), 400

        # If the node is not in any pipeline, proceed with deletion
        Node.delete(node_id=node.get('node_id'))
        return jsonify({"response": "success", "message": "Node deleted successfully!"}), 200
    else:
        return jsonify({"error": f"Unable to delete node with id {node_id}"}), 501

