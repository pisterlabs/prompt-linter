from flask import Flask, request
import json
from flask_cors import CORS
import os
import uuid
from cohere_lib import CoHereClient
import threading

app = Flask(__name__)
CORS(app)
client = CoHereClient(os.environ["API_TOKEN"])


@app.get('/')
def index():
    return "Hello World"


@app.get('/api/new-guide')
def new_guide():
    prompt = request.args.get("prompt")
    id = uuid.uuid4()

    cohere = threading.Thread(target=client.save_guide_text, args=(id, prompt))
    cohere.start()
    return {
        'id': id
    }


@app.get('/api/guide/<id>')
def get_guide(id):
    directory = os.path.join(os.environ["GUIDE_DIRECTORY"], f'{id}\\output.json')
    if not os.path.exists(directory):
        return {
            'status': 404,
            'error': 'Guide not found'
        }

    with open(os.path.join(os.environ["GUIDE_DIRECTORY"], f'{id}\\output.json'), 'r') as outfile:
        return {
            'status': 200,
            'response': json.load(outfile)
        }


@app.get('/api/guide/<id>/<image>')
def get_image(id, image):
    path = os.path.join(os.environ["GUIDE_DIRECTORY"], f'{id}\\images\\${image}')
    if not os.path.exists(path):
        return {
            'status': 404,
            'error': 'Image not found'
        }

    return app.send_static_file(path)


if __name__ == '__main__':
    print("Starting server...")
    app.run(threaded=True)
