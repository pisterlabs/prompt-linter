import slack
from flask import Flask
from slackeventsapi import SlackEventAdapter

# from slack.errors import SlackApiError
import requests

# from load_db_start_chat import create_qa, qa_complete, create_logger
from dotenv import load_dotenv
import os
import openai
import uuid
import time

SLACK_TOKEN = "xoxb-5608865541751-5647334618240-E0YDj983gFoxGq7XCCubTeMY"
SIGNING_SECRET = "286780ebbbf56d06873dcae941d34500"

# app = Flask(__name__)
# slack_event_adapter = SlackEventAdapter(SIGNING_SECRET, "/slack/events", app)

# client = slack.WebClient(token=SLACK_TOKEN)
# #ai-product

app = Flask(__name__)
slack_event_adapter = SlackEventAdapter(SIGNING_SECRET, "/slack/events", app)

client = slack.WebClient(token=SLACK_TOKEN)
# credential_path = "../.credential"
# load_dotenv(credential_path)

# openai.api_key = os.environ["OPENAI_API_KEY"]

from modules.chatbtl_utils import (
    create_qa,
    qa_complete,
    create_logger,
    load_db,
    get_memory,
    load_chat_memory,
)

# logger = create_logger()
# qa = create_qa()

chat_history = []
chat_history_token = []
prev_event_id = None
dir_path = os.path.dirname(os.path.realpath(__file__))

db_folder = f"{dir_path}/../db_folder"


def qa_slack_complete(text):
    product_id = "product_a"
    product_id = "product_a_small"
    client_id = "slack"
    request_id = uuid.uuid4().hex
    chat_session_id = "chat_session_id_1"
    start_time = time.time()
    db_path = os.path.join(db_folder, product_id)
    print("db_path: {}".format(db_path))
    db_loaded = load_db(db_path)
    memory = get_memory(
        client_id=client_id,
        product_id=product_id,
        chat_session_id=chat_session_id,
    )

    qa = create_qa(db=db_loaded, memory=memory, db_params={})
    print("load db time: {}".format(time.time() - start_time))
    print("db_loaded: {}".format(db_loaded.get()))
    logger = create_logger(
        client_id=client_id,
        product_id=product_id,
        chat_session_id=chat_session_id,
    )

    (
        response_text,
        source_documents_metadatas,
    ) = qa_complete(
        qa,
        text,
        logger=logger,
        client_id=client_id,
        product_id=product_id,
        chat_session_id=chat_session_id,
    )
    print("source_documents_metadatas", source_documents_metadatas)
    print("response_text", response_text)
    response_text = f"{response_text}\n\n{source_documents_metadatas}"
    return response_text


chat_api_url = "http://localhost:9300"


@slack_event_adapter.on("message")
def message(payload):
    global prev_event_id
    print(payload)
    event = payload.get("event", {})
    channel_id = event.get("channel")
    user_id = event.get("user")
    text = event.get("text")
    # Check if the event_id is the same as the previous one,
    # If it is, skip processing to prevent duplicate responses
    current_event_id = payload.get("event_id")
    if current_event_id == prev_event_id:
        return

    # Set the current event_id as the prev_event_id for the next iteration
    prev_event_id = current_event_id

    # bot_id = "U05K19UJ672"
    if user_id == "U05K19UJ672":
        print("\n\nbot message\n\n")
        return
    data = {"message": text}
    product_id = "product_a"
    client_id = "slack"
    chat_session_id = "session_1"
    headers = {
        "X-Client-ID": f"{client_id}",
        "X-Product-ID": f"{product_id}",
        "X-Chat-Session-ID": f"{chat_session_id}",
    }
    response = requests.post(
        f"{chat_api_url}/chat",
        json=data,
        headers=headers,
    )
    try:
        response_dict = response.json()
        source_data_list = response_dict["data"]["source_data"]
        source_data_list
        response_text = response_dict["data"]["message"] + "\nSource data:"

        for source_data in source_data_list:
            response_text += f"\n   {source_data}"
        print(response_text)
    except:
        response_text = "failed to get response from chat api"
    print(f"\n\t{response_text}")
    client.chat_postMessage(channel=channel_id, text=response_text)

    # if text == "hi":
    #     client.chat_postMessage(channel=channel_id, text="Hello")
    # if text == "image":
    #     try:
    #         response = client.files_upload(
    #             file="/home/pragnakalpdev23/mysite/slack_file_display/download (2).jpg",
    #             initial_comment="This is a sample Image",
    #             channels=channel_id,
    #         )
    #     except SlackApiError as e:
    #         # You will get a SlackApiError if "ok" is False
    #         assert e.response["ok"] is False
    #         # str like 'invalid_auth', 'channel_not_found'
    #         assert e.response["error"]
    #         print(f"Got an error: {e.response['error']}")
    # if text == "video":
    #     try:
    #         response = client.files_upload(
    #             file="/home/pragnakalpdev23/mysite/slack_file_display/sample-mp4-file-small.mp4",
    #             # initial_comment='This is a sample video',
    #             channels=channel_id,
    #         )
    #     except SlackApiError as e:
    #         # You will get a SlackApiError if "ok" is False
    #         assert e.response["ok"] is False
    #         # str like 'invalid_auth', 'channel_not_found'
    #         assert e.response["error"]
    #         print(f"Got an error: {e.response['error']}")
    # if text == "file":
    #     try:
    #         response = client.files_upload(
    #             file="/home/pragnakalpdev23/mysite/slack_file_display/sample.pdf",
    #             # initial_comment='This is a sample file',
    #             channels=channel_id,
    #         )
    #     except SlackApiError as e:
    #         # You will get a SlackApiError if "ok" is False
    #         assert e.response["ok"] is False
    #         # str like 'invalid_auth', 'channel_not_found'
    #         assert e.response["error"]
    #         print(f"Got an error: {e.response['error']}")


if __name__ == "__main__":
    app.run(debug=True, port=3000, use_reloader=False)
