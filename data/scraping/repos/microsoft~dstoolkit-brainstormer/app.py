# https://github.com/MariyaSha/greetingApp_FLASK/blob/main/templates/index.html
# https://www.youtube.com/watch?v=6plVs_ytIH8
import pandas as pd
import json
import openai
from flask import Flask, render_template, request, flash, jsonify
import os
import random
import logging
from dotenv import load_dotenv
from llm_utils import LLMUtils
from pathlib import Path
from datetime import datetime

# --set open ai key
load_dotenv()
openai.api_type = "azure"
openai.api_base = os.getenv("api_base")
openai.api_version = "2023-03-15-preview"
openai.api_key = os.getenv("api_key")

app = Flask(__name__)
app.debug = True

# ---logging function setting
curr_logfilename = Path("logs") / f"{datetime.now():%Y-%m-%d-%H-%M-%S%z}.log"
if not curr_logfilename.parent.exists():
    curr_logfilename.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=str(curr_logfilename),
    format="=%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
llm_utils = LLMUtils(
    document_parquet_path=Path("assets/ip_description_topic_embedding.parquet"),
    llm_config_path=Path("assets/llm_config.json"),
    raw_doc_path=Path("assets/ip_description.json"),
)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat")
def render_chat():
    return render_template("chat.html")


@app.route("/idea", methods=['POST'])
def ideation_stage():
    if request.method == 'POST':
        user_input = request.json['user_input']
        usecase_number = request.json['usecase_number']
        return llm_utils.ideation(user_input=user_input, usecase_number=usecase_number)
    else:
        return str("Only POST request allowed")


@app.route("/ip", methods=['POST'])
def ip_recommendation():
    if request.method == 'POST':
        original_input = request.json['original_input']
        generated_idea = request.json['use_case_list']
        return llm_utils.recommend_ips(original_input=original_input, generated_idea=generated_idea)
    else:
        return str("Only POST request allowed")


@app.route("/get_random_idea", methods=['GET'])
def get_random_idea():
    with open('assets/test_case.json', 'r') as f:
        idea_list = json.load(f)
        total_idea = len(idea_list)
        random_id = random.randrange(0,total_idea)
        random_idea = idea_list[random_id]
        return jsonify(random_idea)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
