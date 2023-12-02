import os
import uuid
import traceback

from dotenv import load_dotenv
from flask import Flask, request, session, jsonify, current_app
from langchain.agents import create_csv_agent
from langchain.llms import OpenAI
from waitress import serve
import asyncio
import re

secret_key = uuid.uuid4().hex

app = Flask(__name__)
print('running', flush=True)
app.secret_key = secret_key

app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, )
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024


@app.route("/api/test")
def test():
    return "<p>test<p>"


# FIXME: Currently, this is dangerous as there is no security between different sessions
@app.route("/api/upload-csv", methods=['POST'])
def receive_file():
    if "file" not in request.files:
        return "No file part", 400

    file = request.files["file"]

    if file.filename == "":
        return "No selected file", 400

    unique_id = str(uuid.uuid4())

    temp_dir = os.path.join(os.path.dirname(__file__), "temp_files", unique_id)
    os.makedirs(temp_dir, exist_ok=True)

    file_path = os.path.join(temp_dir, file.filename)
    session['file_path'] = file_path
    session['agent'] = None
    file.save(file_path)

    load_dotenv()
    os.getenv("OPENAI_API_KEY")

    current_app.agent = create_csv_agent(
        OpenAI(temperature=0),
        file_path,
        max_iterations=10000,
        verbose=True
    )
    print("agent created", flush=True)

    return "received", 200


@app.route("/api/ask-gpt", methods=['POST'])
def ask_gpt():
    if request.json['question'] == 'test':
        return jsonify(
            answer='testing alot of words testing alot of words testing alot of words testing alot of words testing alot of words testing alot of words testing alot of words testing alot of words testing alot of words ')

    agent = current_app.agent

    content = request.json
    content = content['question']
    print(content, flush=True)

    if content is not None and content != "":
        return jsonify(answer=agent.run(content))
    else:
        return "Invalid question"

@app.route("/api/get-map", methods=['POST'])
def get_map():
    content = request.json['question']
    agent = current_app.agent

    map_response = agent.run(content)
    print(map_response, flush=True)

    try:
        # Use regular expression to find latitude and longitude in the input text
        # First, try to find the format "34.149917 latitude and -118.616783 longitude."
        pattern1 = r'(-?\d+\.\d+)\s*latitude\s*and\s*(-?\d+\.\d+)\s*longitude\.'
        match1 = re.search(pattern1, map_response)

        # If the first format is not found, try to find the format "34.10155, -118.336883."
        if not match1:
            pattern2 = r'(-?\d+\.\d+),\s*(-?\d+\.\d+)'
            match2 = re.search(pattern2, map_response)
            if not match2:
                raise ValueError
            lat, lon = map(float, match2.groups())
        else:
            lat, lon = map(float, match1.groups())

        return jsonify(latitude=lat, longitude=lon)

    except Exception as e:
        traceback.print_exc()
        return "Error extracting coordinates. Please enter a valid input."


if __name__ == "__main__":
    # app.run(port=5328) only use for development
    serve(app, port=5328)

