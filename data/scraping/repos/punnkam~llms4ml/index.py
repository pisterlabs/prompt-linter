from flask import Flask, request, redirect, url_for
from flask_cors import CORS, cross_origin
import boto3
import os
import io
import shortuuid
import re
import json
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)
s3 = boto3.client(
    "s3",
    aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.environ.get("AWS_ACCESS_KEY_SECRET"),
)

from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT

anthropic = Anthropic()  # defaults to os.environ.get("ANTHROPIC_API_KEY")


def find_largest_array(s):
    # Find all arrays in the string
    arrays = re.findall(r"\[([^]]*)\]", s)

    if not arrays:
        return None

    # Split each array string into a list of elements
    arrays = [arr.split(",") for arr in arrays]

    # Find the array with the most elements
    largest_array = max(arrays, key=len)

    # Convert the array back into a string and return it
    return "[" + ",".join(largest_array) + "]"


def validate_file(file):
    # Check if the file is one of the allowed types
    if not file.filename.endswith((".csv", ".txt", ".xml")):
        return "Invalid file type", 400

    # Load the data into a pandas DataFrame
    if file.filename.endswith(".csv") or file.filename.endswith(".txt"):
        df = pd.read_csv(file)
    elif file.filename.endswith(".xml"):
        df = pd.read_xml(file)  # As of pandas 1.3.0, pandas supports reading XML files

    # Columns must be labeled by string
    if not all(isinstance(col, str) for col in df.columns):
        return "All columns must be labeled by string", 400

    # Check if the DataFrame has at least 5 rows
    if df.shape[0] < 5:
        return "The file must contain at least 5 rows of data", 400

    return "File is valid", 200


def read_file_from_s3(filename):
    obj = s3.get_object(Bucket="ml-llms", Key=filename)
    return obj["Body"].read().decode("utf-8")


@app.route("/")
def index():
    return "Hello World!"


@app.route("/classification", methods=["POST"])
def call_classify():
    body = request.get_json()
    print(body)
    filename = body["filename"]
    prompt = read_file_from_s3(filename)
    prompt += f"""\nThe data to predict is:\n{body["data"]}"""
    prompt += "\nIf you have the answer, return an array of labels for each missing label denoted by an underscore '_'."

    completion = anthropic.completions.create(
        model="claude-2",
        max_tokens_to_sample=10000,
        prompt=f"{HUMAN_PROMPT} {prompt} {AI_PROMPT}",
    )
    print(completion.completion)

    return find_largest_array(completion.completion), 200


@app.route("/regression", methods=["POST"])
def call_regression():
    body = request.get_json()
    print(body)
    filename = body["filename"]
    prompt = read_file_from_s3(filename)
    prompt += f"""\nThe data to predict is:\n{body["data"]}"""
    prompt += "\nIf you have the answer, return an array of labels for each missing label denoted by an underscore '_'."

    completion = anthropic.completions.create(
        model="claude-2",
        max_tokens_to_sample=10000,
        prompt=f"{HUMAN_PROMPT} {prompt} {AI_PROMPT}",
    )
    print(completion.completion)

    return find_largest_array(completion.completion), 200


def get_regression_prompt(data, form):
    task = ""
    predictive_var = ""
    for item in form:
        if item["label"] == "Task":
            task = item["value"]
        elif item["label"] == "Predictive Variable":
            labels = item["value"]

    prompt = f"You are RegressionAI. You will be provided some data below and you need to predict the value of the data. "
    prompt += f"The regression task is: {task}. You must determine the value(s) for {labels}. A missing label is indicated by an underscore '_'."
    prompt += """\nPlease think step by step following this:
    1. Comprehend the data and the task.
    2. Determine the features that are important for the task.
    3. If you can't determine the label, return [].
    4. Determine a label for each feature with an underscore '_'.
    """
    prompt += f"""\nThe available data is: \n{data}"""

    return prompt


def get_classify_prompt(data, form):
    task = ""
    num_classes = ""
    labels = ""
    for item in form:
        if item["label"] == "Task":
            task = item["value"]
        elif item["label"] == "Number of Classes":
            num_classes = item["value"]
        elif item["label"] == "Labels (comma-separated)":
            labels = item["value"]

    prompt = f"You are ClassifyAI. You will be provided some data below and you need to predict the class of the data. "
    prompt += f"The classification task is: {task}. You must determine the label(s) for {labels}. A missing label is indicated by an underscore '_'."
    prompt += """\nPlease think step by step following this:
    1. Comprehend the data and the task.
    2. Determine the features that are important for the task.
    3. If you can't determine the label, return [].
    4. Determine a label for each feature with an underscore '_'.
    """
    prompt += f"""\nThe available data is: \n{data}"""

    return prompt


@app.route("/upload", methods=["POST"])
def upload_file_to_s3():
    if "file" not in request.files:
        return "No 'file' key in request.files", 400

    file = request.files["file"]
    data = json.loads(request.form.get("json"))

    # Read file contents into a BytesIO object
    file.seek(0)
    file_content = io.BytesIO(file.read())
    file.seek(0)  # Reset file pointer

    # Create a copy of the file for upload
    file_copy = io.BytesIO(file_content.getvalue())

    text_content = file_copy.read().decode("utf-8")
    model = request.args.get("model")

    if model == "classification":
        text_content = get_classify_prompt(text_content, data)
    elif model == "regression":
        pass
    elif model == "recommendation":
        pass

    new_file = io.BytesIO(text_content.encode("utf-8"))

    msg, status_code = validate_file(file)
    print(msg, status_code)
    if status_code != 200:
        return msg, status_code

    if file.filename == "":
        return "No selected file", 400

    newName = shortuuid.uuid()

    try:
        s3.upload_fileobj(new_file, "ml-llms", newName)
    except Exception as e:
        return str(e), 500

    return newName, 200
