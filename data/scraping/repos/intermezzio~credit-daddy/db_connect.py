import time
import os
import math
import heapq
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from icecream import ic

from cohere_script import float_def

# Initialize Firebase
try:
    cred = credentials.Certificate("firebase-sdk-admin.json")
    firebase_admin.initialize_app(cred)
except:
    firebase_admin.initialize_app(
        credential=credentials.Certificate(
            {
                "type": "service_account",
                "project_id": os.environ.get("FIREBASE_PROJECT_ID"),
                "private_key_id": os.environ.get("PRIVATE_KEY_ID"),
                "private_key": os.environ.get("FIREBASE_PRIVATE_KEY").replace(
                    "\\n", "\n"
                ),
                "client_email": os.environ.get("FIREBASE_CLIENT_EMAIL"),
                "client_id": os.environ.get("CLIENT_ID"),
                "auth_uri": os.environ.get("AUTH_URI"),
                "token_uri": os.environ.get("TOKEN_URI"),
                "auth_provider_x509_cert_url": os.environ.get(
                    "AUTH_PROVIDER_X509_CERT_URL"
                ),
                "client_x509_cert_url": os.environ.get("CLIENT_X509_CERT_URL"),
            }
        ),
    )

# Initialize Firestore Database
db = firestore.client()

# env variables
load_dotenv()

api_key = os.getenv("FIREBASE_API_KEY")


def name_to_id(name: str):
    return name.lower().replace(" ", "-")


def upload_card(
    name: str,
    company_name: str,
    card_type: str,
    avg_apr: float,
    min_cashback: float,
    max_cashback: float,
    foreign_fee: float,
    intro_offer_details: str,
    annual_fee: float,
    overcharge_fee: float,
    contract: str = ""
):
    id_ = name_to_id(name)
    row_data = {
        "id": id_,
        "name": name,
        "company-name": company_name,
        "card-type": card_type,
        "avg-apr": avg_apr,
        "min-cashback": min_cashback,
        "max-cashback": max_cashback,
        "foreign-fee": foreign_fee,
        "intro-offer-details": intro_offer_details,
        "annual-fee": annual_fee,
        "overcharge-fee": overcharge_fee,
        "contract": contract,
    }

    if not get_card(id_):
        db.collection("cards").add(row_data)

    return id_


def upload_chat(id_: str, question: str, answer: str):
    row_data = {
        "id": id_,
        "question": question,
        "answer": answer,
        "timestamp": time.time(),
    }
    db.collection("questions").add(row_data)


def get_card(id_: str):
    query = db.collection("cards").where(filter=firestore.FieldFilter("id", "==", id_))
    cards = query.stream()

    try:
        if card := next(cards):
            return card.to_dict()
    except StopIteration:
        return None


def get_chats(id_: str):
    query = (
        db.collection("questions")
        .where(filter=firestore.FieldFilter("id", "==", id_))
        .order_by("timestamp")
    )

    return [chat.to_dict() for chat in query.stream()]


def get_contract(id_: str) -> str:
    if card := get_card(id_):
        return card.get("contract", "")
    return ""

def calc_metric(
    foreign_overcharge: float,
    apr_annual: float,
    selective_general: float,
    row_data: dict,
):
    card_score = (
        float_def(row_data["foreign-fee"],0) / 3 * (1 - foreign_overcharge)
        + float_def(row_data["overcharge-fee"] or 30, 30) / 15 * foreign_overcharge
        + float_def(row_data["avg-apr"], 20) / 15 * (1 - apr_annual)
        - math.log(float_def(row_data["annual-fee"], 0) + 1) / 2 * apr_annual
        + float_def(row_data["min-cashback"], 0) * (1 - selective_general)
        + float_def(row_data["max-cashback"], 0) / 3 * selective_general
    )

    return card_score


def get_optimal(
    foreign_overcharge: float,
    apr_annual: float,
    selective_general: float,
    n: int,
):
    id_to_metric = {}

    for card in db.collection("cards").stream():
        card_dict = card.to_dict()
        
        id_to_metric.update({
            card_dict["id"]: calc_metric(
                foreign_overcharge, apr_annual, selective_general, card_dict
            ),
        })

    ic(id_to_metric)

    return heapq.nlargest(n, id_to_metric.keys(), key=id_to_metric.get)
