import re
import uuid

from dotenv import find_dotenv, load_dotenv
from flask import Flask, request, jsonify
from langchain.callbacks.tracers import langchain

from advisor.agents import init_convo_agent

langchain.debug = True

app = Flask(__name__)


def initialize():
    load_dotenv(find_dotenv())
    app.config['agents_map'] = {}


@app.route('/advice', methods=['POST'])
def give_advice():
    data = request.get_json()
    convo_id = data.get('convo_id', '')
    question = data.get('question', '')

    if not convo_id:
        convo_id = str(uuid.uuid4())

    if convo_id not in app.config['agents_map']:
        app.config['agents_map'][convo_id] = init_convo_agent(convo_id)

    agent = app.config['agents_map'][convo_id]
    answer = agent.run(question)

    # The ?<= in a regular expression is a special construct called a positive lookbehind assertion.
    # It is used to specify a condition that must be met before the main pattern is matched,
    # but the text that matches the lookbehind assertion is not included in the match
    restaurant_ids = re.findall(r'(?<=##: )[\w,]+', answer)
    answer = re.sub(r'##: [\w,]+$', '', answer)

    response_data = {'convo_id': convo_id, 'answer': answer, 'restaurant_ids': restaurant_ids}
    return jsonify(response_data)


@app.route('/convo/<convo_id>', methods=['DELETE'])
def evict_agent(convo_id):
    if convo_id in app.config['agents_map']:
        del app.config['agents_map'][convo_id]
        return jsonify({"message": f"Agent for convo_id {convo_id} evicted."}), 200
    else:
        return jsonify({"message": f"No agent found for convo_id {convo_id}."}), 404


if __name__ == '__main__':
    initialize()
    app.run(debug=True, port=8080)
