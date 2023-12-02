# from http.server import HTTPServer, BaseHTTPRequestHandler
# from urllib.parse import parse_qs
import json
import os
import sys

from flask import Flask, redirect, url_for, request, make_response
from flask_cors import CORS, cross_origin
from icecream import ic

from werkzeug.utils import secure_filename

import cohere_script
import db_connect
from PDFReader import pdf_to_text


UPLOAD_FOLDER = "../cache"
ALLOWED_EXTENSIONS = {"pdf", "txt"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
CORS(app)

@app.route("/card/<id_>", methods=["GET"])
def get_card(id_):
    # get card info
    card_info = db_connect.get_card(id_)
    return card_info


@app.route("/chats/<id_>", methods=["GET"])
def get_chats(id_):
    # get chats
    chats = db_connect.get_chats(id_)
    return chats


@app.route("/ask/<id_>/<question>", methods=["GET"])
def get_answer(id_, question):
    contract = ic(db_connect.get_contract(id_))
    answer = cohere_script.ask_more_card_details(id_, question)

    db_connect.upload_chat(id_, question, answer)

    response = make_response(answer, 200)
    response.mimetype = "text/plain"
    return response


@app.route("/upload-card", methods=["POST"])
def analyze_contract():
    if "file" not in request.files:
        pass

    file = request.files["file"]
    if file.filename.rsplit(".", 1)[1].lower() not in ALLOWED_EXTENSIONS:
        pass

    filename = secure_filename(file.filename)
    abs_filename = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(abs_filename)

    # upload pdf and convert to text
    if abs_filename.endswith(".pdf"):
        contract_text = pdf_to_text(abs_filename)
    elif abs_filename.endswith(".txt"):
        with open(abs_filename) as infile:
            contract_text = infile.read().strip()

    # text to results
    card_details = cohere_script.extract_card_details(contract_text)

    # store results in firebase
    id_ = db_connect.upload_card(**card_details)

    # return new id
    response = make_response(f"http://www.creditdaddy.tech/tldr.html?card={id_}", 200)
    response.mimetype = "text/plain"
    return response


@app.route("/compare/<id_>", methods=["GET"])
def compare_card(id_):
    model_cards = ["td-cash-back-visa-infinite", "cibc-aventura-visa-infinite-privilege",
        "rbc-visa-classic-low-rate", "cibc-dividend-visa", "TD-Regular-CreditCard"]
    
    if id_ not in model_cards:
        ids = [id_] + model_cards[:4]
    else:
        other_model_cards = set(model_cards)
        other_model_cards.remove(id_)
        ids = [id_] + list(other_model_cards)
    
    response = make_response(f"http://www.creditdaddy.tech/card-compare.html?card={','.join(ids)}", 200)
    response.mimetype = "text/plain"
    return response
    

@app.route("/recommend", methods=["GET"])
def recommend_cards(
    foreign_overcharge: float = 0,
    apr_annual: float = 0, # apr_annual
    selective_general: float = 0, # selective_general cashback
    n: int = 5,
):
    ids = db_connect.get_optimal(foreign_overcharge, apr_annual, selective_general, n)

    if n == 1:
        # return new id
        response = make_response(f"http://www.creditdaddy.tech/tldr.html?card={ids[0]}", 200)
        response.mimetype = "text/plain"
        return response
    else:
        # return new id
        response = make_response(f"http://www.creditdaddy.tech/card-compare.html?card={','.join(ids)}", 200)
        response.mimetype = "text/plain"
        return response

if __name__ == "__main__":
    app.run(debug=True, port=3000)
