import os
import openai
import datetime
import time
from bson.objectid import ObjectId
from flask import Blueprint, render_template, request, Response, jsonify, stream_with_context, g
from ..db_utils import mongodb_connect
from ...API.account.account_verification_api import check_verification
from dotenv import load_dotenv



db = mongodb_connect()

save_result_api_blueprint = Blueprint('save_result_api', __name__, url_prefix='/API/save_result')

# This Route: /API/save_result
@save_result_api_blueprint.route('/', methods=['POST'])
@check_verification(['user','admin'])
def save_result():
    data = request.json
    policy_name = data.get("policy_name", "")
    serverId = data.get("serverId", "")
    result = data.get("option", "")
    result_comment = data.get("answer", "")
    servername = data.get("servername","")
    print(data)

    db.log.insert_one(
        {
            "log_type" : "Security Diagnostics",
            "user_id" : g.user_id,
            "policy_name" : policy_name,
            "server name" : servername,
            "result" : result,
            "result_comment" : result_comment
        }
    )




    print("success")
    return jsonify({"status": "success"})

# db.log.insert_one(
#             {
#                 "log_type": "Linux Security Assistant",
#                 "user_id": g.user_id,
#                 "question" : message,
#                 "answer" : total_answer,
#                 "time" : (datetime.datetime.utcnow() + datetime.timedelta(hours=9)).strftime('%Y-%m-%d %H:%M:%S')
#             }
#         )
